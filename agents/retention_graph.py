import json
from functools import lru_cache
from typing import Literal

from langchain.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agents.retention_tools import RETENTION_TOOLS
from schema.agent_state import RetentionState
from schema.retention_models import RetentionPlan
from services.llm_client import get_llm


AGENT_SYSTEM_PROMPT = """
You are a customer retention decision agent for a SaaS company.

The churn probability has already been produced by an XGBoost model.
DiCE has already generated counterfactual alternatives.

Your job is not to replace the ML model or DiCE.

Your job is to decide which model-based alternatives are operationally
realistic and which business actions should follow.

For every tool call, include a decision_reason.

decision_reason is displayed in the user interface. It should briefly
explain why the specific tool is needed at the current point in the
decision process.

Good example:

"Option 1 requires a large fee reduction. I need to verify whether that
discount is permitted before treating the option as feasible."

Bad example:

"I need to use the fee tool."

Keep decision_reason to one or two sentences. Do not provide hidden
chain-of-thought. Only give a concise user-facing explanation of the
immediate business decision being evaluated.

Required behaviour:

1. Inspect the customer using get_customer_context.

2. Inspect the churn prediction and every available DiCE option using
   get_churn_analysis.

3. Compare all available DiCE options. Do not automatically select
   Option 1.

4. Evaluate each counterfactual based on:
   - predicted churn probability
   - operational feasibility
   - business cost
   - customer experience
   - whether the business can actually control the proposed feature

5. If an option changes monthly_fee, use evaluate_fee_offer before
   treating that option as feasible.

6. If an option changes contract_type, use check_contract_change before
   treating that option as feasible.

7. Some counterfactual features are outcomes or behavioural targets,
   not directly editable business controls.

   Examples:
   - csat_score
   - monthly_logins
   - weekly_active_days

   Interpret these as signals about what the company should try to
   improve. Never pretend the company can directly edit those values.

8. Create a CSM ticket when poor satisfaction, support problems,
   escalations, or account issues justify human intervention.

9. Prepare a customer email only when communication is useful.

10. Do not create an action merely because a tool exists.

11. Use get_case_actions when necessary to avoid duplicate actions.

12. save_email_draft creates a draft only. Never claim that an email
    was sent.

13. Customer-facing actions require human approval.

14. DiCE counterfactuals are model-based explanations. They do not prove
    that the proposed action will causally reduce churn.

15. Never claim that retention is guaranteed.

Once you have gathered enough information and do not require more tools,
respond with a concise strategy summary.

A separate finalize_plan node will convert the completed graph state
into the final structured RetentionPlan.
"""


FINALIZER_SYSTEM_PROMPT = """
You are the finalization node of a LangGraph customer-retention system.

You receive:

- the original customer profile
- the XGBoost churn prediction
- every DiCE counterfactual
- the complete agent and tool conversation

Produce the final RetentionPlan.

Rules:

1. Assess every available DiCE option.

2. At most one option may have decision="selected".

3. It is valid to select no counterfactual if none is operationally
   suitable.

4. expected_churn_probability_after must exactly correspond to the
   selected DiCE option. Never invent a probability.

5. Explain why the chosen strategy was preferred over the alternatives.

6. Base feasibility claims on actual tool results.

7. If an email draft was created, report it as "prepared", never "sent".

8. If a CSM ticket was created, report it as "created".

9. Customer-facing actions require human approval.

10. Include the limitation that DiCE recommendations are model-based
    counterfactuals and are not causal guarantees.

11. Do not invent tool IDs, business policies, actions, or model results.
"""


TOOL_PURPOSES = {
    "get_customer_context": "Inspect customer profile and account context.",
    "get_churn_analysis": "Inspect churn risk and all DiCE alternatives.",
    "get_case_actions": "Check previously created retention actions.",
    "evaluate_fee_offer": "Validate a fee reduction against retention policy.",
    "check_contract_change": "Validate a proposed contract transition.",
    "save_email_draft": "Prepare customer communication without sending it.",
    "create_csm_ticket": "Create an internal Customer Success follow-up.",
}


checkpointer = InMemorySaver()


def initialize_case_node(state: RetentionState):
    writer = get_stream_writer()

    prediction = state["prediction_payload"]
    interventions = prediction.get("interventions", [])
    probability = prediction.get("churn_probability")

    writer(
        {
            "event": "node_started",
            "node": "initialize_case",
            "message": "Initializing the retention case.",
        }
    )

    message = HumanMessage(
        content=(
            f"Evaluate this retention case. Current churn probability: "
            f"{probability}. There are {len(interventions)} DiCE alternatives. "
            "Inspect the customer and all counterfactuals before choosing a strategy."
        )
    )

    writer(
        {
            "event": "node_completed",
            "node": "initialize_case",
            "message": "Retention case initialized.",
            "data": {
                "churn_probability": probability,
                "counterfactual_count": len(interventions),
            },
        }
    )

    return {
        "messages": [message],
        "llm_calls": 0,
        "status": "initialized",
        "final_plan": None,
    }


def build_agent_node(model_with_tools):
    def agent_node(state: RetentionState):
        writer = get_stream_writer()

        writer(
            {
                "event": "node_started",
                "node": "agent",
                "message": "Agent is evaluating the current graph state.",
            }
        )

        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            *state.get("messages", []),
        ]

        response = model_with_tools.invoke(messages)
        tool_calls = response.tool_calls or []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            args = dict(tool_call.get("args", {}))

            decision_reason = args.pop("decision_reason", None)

            writer(
                {
                    "event": "decision",
                    "node": "agent",
                    "tool": tool_name,
                    "purpose": TOOL_PURPOSES.get(
                        tool_name,
                        "Agent-selected business tool.",
                    ),
                    "decision_reason": decision_reason,
                    "arguments": args,
                }
            )

        if tool_calls:
            status = "tool_requested"
            message = f"Agent requested {len(tool_calls)} tool call(s)."
        else:
            status = "ready_to_finalize"
            message = "Agent has enough information to finalize the plan."

        writer(
            {
                "event": "node_completed",
                "node": "agent",
                "message": message,
            }
        )

        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
            "status": status,
        }

    return agent_node


def route_after_agent(
    state: RetentionState,
) -> Literal["tools", "finalize_plan"]:
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return "finalize_plan"


def build_finalize_node(finalizer_model):
    def finalize_plan_node(state: RetentionState):
        writer = get_stream_writer()

        writer(
            {
                "event": "node_started",
                "node": "finalize_plan",
                "message": "Creating the final structured retention plan.",
            }
        )

        source_data = {
            "customer_profile": state["customer_profile"],
            "prediction_payload": state["prediction_payload"],
        }

        source_message = HumanMessage(
            content=(
                "Use the following original application data as source truth:\n\n"
                + json.dumps(source_data, indent=2, default=str)
            )
        )

        messages = [
            SystemMessage(content=FINALIZER_SYSTEM_PROMPT),
            *state.get("messages", []),
            source_message,
        ]

        plan = finalizer_model.invoke(messages)

        writer(
            {
                "event": "node_completed",
                "node": "finalize_plan",
                "message": "Structured retention plan created.",
                "data": {
                    "selected_counterfactual_id": plan.selected_counterfactual_id,
                    "requires_human_approval": plan.requires_human_approval,
                },
            }
        )

        return {
            "final_plan": plan,
            "status": "completed",
        }

    return finalize_plan_node


@lru_cache(maxsize=1)
def get_retention_graph():
    llm = get_llm(temperature=0.2)

    model_with_tools = llm.bind_tools(
        RETENTION_TOOLS,
        strict=True,
    )

    finalizer_model = llm.with_structured_output(
        RetentionPlan,
        method="function_calling",
    )

    agent_node = build_agent_node(model_with_tools)
    finalize_node = build_finalize_node(finalizer_model)
    tools_node = ToolNode(RETENTION_TOOLS)

    workflow = StateGraph(RetentionState)

    workflow.add_node("initialize_case", initialize_case_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("finalize_plan", finalize_node)

    workflow.add_edge(START, "initialize_case")
    workflow.add_edge("initialize_case", "agent")

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "finalize_plan": "finalize_plan",
        },
    )

    workflow.add_edge("tools", "agent")
    workflow.add_edge("finalize_plan", END)

    return workflow.compile(checkpointer=checkpointer)