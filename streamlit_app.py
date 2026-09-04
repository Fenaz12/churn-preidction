import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
)

PREDICT_URL = f"{BACKEND_URL}/predict"
AGENT_STREAM_URL = f"{BACKEND_URL}/agent/orchestrate/stream"

REQUEST_TIMEOUT = 120


NODE_UI = {
    "initialize_case": {
        "name": "Initialize Case",
        "purpose": (
            "Creates the initial LangGraph state containing the customer, "
            "prediction, and DiCE alternatives."
        ),
    },
    "agent": {
        "name": "Agent Node",
        "purpose": (
            "The LLM examines the current state and decides whether "
            "another business tool is required."
        ),
    },
    "tools": {
        "name": "Tool Node",
        "purpose": (
            "Executes the tools selected by the LLM and returns their "
            "results to the shared state."
        ),
    },
    "finalize_plan": {
        "name": "Finalize Plan",
        "purpose": (
            "Converts the completed agent and tool conversation into "
            "a structured retention plan."
        ),
    },
}


TOOL_UI = {
    "get_customer_context": {
        "name": "Inspect Customer",
        "purpose": (
            "Reads the customer's account, subscription, usage, support, "
            "and satisfaction information."
        ),
    },
    "get_churn_analysis": {
        "name": "Compare Churn Alternatives",
        "purpose": (
            "Loads the XGBoost churn prediction and every DiCE alternative."
        ),
    },
    "get_case_actions": {
        "name": "Check Existing Actions",
        "purpose": (
            "Checks whether this case already has tickets or drafts."
        ),
    },
    "evaluate_fee_offer": {
        "name": "Validate Fee Offer",
        "purpose": (
            "Checks whether a proposed fee reduction follows retention "
            "policy and estimates its cost."
        ),
    },
    "check_contract_change": {
        "name": "Validate Contract Change",
        "purpose": (
            "Checks whether a proposed contract transition is valid."
        ),
    },
    "save_email_draft": {
        "name": "Prepare Customer Email",
        "purpose": (
            "Creates customer communication without actually sending it."
        ),
    },
    "create_csm_ticket": {
        "name": "Create CSM Ticket",
        "purpose": (
            "Creates an internal Customer Success follow-up action."
        ),
    },
}


st.set_page_config(
    page_title="Customer Retention Intelligence",
    layout="wide",
)


def initialize_session_state() -> None:
    defaults = {
        "customer_data": None,
        "prediction_result": None,
        "agent_result": None,
        "agent_events": [],
        "case_id": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_case() -> None:
    st.session_state.customer_data = None
    st.session_state.prediction_result = None
    st.session_state.agent_result = None
    st.session_state.agent_events = []
    st.session_state.case_id = None


def call_api(
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = requests.post(
        url,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )

    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"API returned {response.status_code}: {detail}"
        )

    return response.json()


def stream_agent(payload: dict[str, Any]):
    with requests.post(
        AGENT_STREAM_URL,
        json=payload,
        stream=True,
        timeout=(10, 180),
    ) as response:

        if not response.ok:
            raise RuntimeError(
                f"Agent API returned {response.status_code}: {response.text}"
            )

        for line in response.iter_lines(decode_unicode=True):
            if line:
                yield json.loads(line)


def format_feature_name(feature: str) -> str:
    return str(feature).replace("_", " ").title()


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)


def format_probability(value: float | None) -> str:
    if value is None:
        return "N/A"

    return f"{value * 100:.1f}%"


def get_node_info(node_name: str) -> dict[str, str]:
    return NODE_UI.get(
        node_name,
        {
            "name": node_name,
            "purpose": "LangGraph execution node.",
        },
    )


def get_tool_info(tool_name: str) -> dict[str, str]:
    return TOOL_UI.get(
        tool_name,
        {
            "name": tool_name,
            "purpose": "Agent-selected business tool.",
        },
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.header("System")

        st.caption("Backend")
        st.code(BACKEND_URL)

        st.divider()

        st.subheader("Application Flow")

        st.write("1. XGBoost predicts churn probability.")
        st.write("2. DiCE generates counterfactual alternatives.")
        st.write("3. LangGraph evaluates feasibility.")
        st.write("4. Business tools return constraints.")
        st.write("5. The graph creates a retention plan.")

        with st.expander("LangGraph Structure"):
            st.graphviz_chart(
                """
                digraph {
                    rankdir=TB;

                    START -> initialize_case;
                    initialize_case -> agent;

                    agent -> tools [label="tool requested"];
                    tools -> agent [label="tool result"];

                    agent -> finalize_plan [label="no more tools"];
                    finalize_plan -> END;
                }
                """,
                width="stretch",
            )

        st.divider()

        if st.button("Reset Retention Case", width="stretch"):
            reset_case()
            st.rerun()


def build_customer_form():
    with st.form("customer_form"):
        st.subheader("Customer Profile")

        st.caption(
            "Enter the customer's current business, usage, "
            "support, and engagement information."
        )

        customer_tab, usage_tab, support_tab = st.tabs(
            [
                "Customer",
                "Subscription and Usage",
                "Support and Engagement",
            ]
        )

        with customer_tab:
            col1, col2 = st.columns(2)

            with col1:
                age = st.slider("Age", 18, 100, 30)

                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"],
                )

                country = st.selectbox(
                    "Country",
                    [
                        "Bangladesh",
                        "Canada",
                        "Germany",
                        "Australia",
                        "India",
                        "USA",
                        "UK",
                    ],
                )

                customer_segment = st.selectbox(
                    "Customer Segment",
                    [
                        "Enterprise",
                        "SME",
                        "Individual",
                    ],
                )

            with col2:
                signup_channel = st.selectbox(
                    "Signup Channel",
                    [
                        "Web",
                        "Mobile",
                        "Referral",
                    ],
                )

                discount_applied = st.selectbox(
                    "Discount Applied",
                    ["No", "Yes"],
                )

                price_increase_last_3m = st.selectbox(
                    "Price Increase in Last 3 Months",
                    ["No", "Yes"],
                )

                survey_response = st.selectbox(
                    "Survey Response",
                    [
                        "Satisfied",
                        "Neutral",
                        "Unsatisfied",
                    ],
                )

        with usage_tab:
            col1, col2 = st.columns(2)

            with col1:
                tenure_months = st.slider(
                    "Tenure (Months)",
                    0,
                    120,
                    12,
                )

                contract_type = st.selectbox(
                    "Contract Type",
                    [
                        "Monthly",
                        "Quarterly",
                        "Yearly",
                    ],
                )

                monthly_fee = st.number_input(
                    "Monthly Fee",
                    min_value=0.0,
                    value=50.0,
                    step=1.0,
                )

                total_revenue = st.number_input(
                    "Total Revenue",
                    min_value=0.0,
                    value=600.0,
                    step=10.0,
                )

                monthly_logins = st.slider(
                    "Monthly Logins",
                    0,
                    100,
                    15,
                )

            with col2:
                weekly_active_days = st.slider(
                    "Weekly Active Days",
                    0,
                    7,
                    3,
                )

                avg_session_time = st.number_input(
                    "Average Session Time (Minutes)",
                    min_value=0.0,
                    value=25.0,
                    step=1.0,
                )

                features_used = st.slider(
                    "Features Used",
                    1,
                    15,
                    4,
                )

                usage_growth_rate = st.slider(
                    "Usage Growth Rate",
                    -0.6,
                    0.6,
                    0.0,
                    step=0.01,
                )

                last_login_days_ago = st.slider(
                    "Days Since Last Login",
                    0,
                    80,
                    5,
                )

        with support_tab:
            col1, col2 = st.columns(2)

            with col1:
                support_tickets = st.slider(
                    "Support Tickets",
                    0,
                    20,
                    1,
                )

                escalations = st.slider(
                    "Escalations",
                    0,
                    10,
                    0,
                )

                avg_resolution_time = st.number_input(
                    "Average Resolution Time (Hours)",
                    min_value=0.0,
                    value=12.0,
                    step=1.0,
                )

                csat_score = st.slider(
                    "CSAT Score",
                    1.0,
                    5.0,
                    4.0,
                    step=0.5,
                )

                complaint_type = st.selectbox(
                    "Complaint Type",
                    [
                        "No_Complaint",
                        "Billing",
                        "Technical",
                        "Service",
                    ],
                )

                nps_score = st.slider(
                    "NPS Score",
                    -100,
                    100,
                    20,
                )

            with col2:
                payment_method = st.selectbox(
                    "Payment Method",
                    [
                        "Card",
                        "PayPal",
                        "Bank_Transfer",
                    ],
                )

                payment_failures = st.slider(
                    "Payment Failures",
                    0,
                    5,
                    0,
                )

                email_open_rate = st.slider(
                    "Email Open Rate",
                    0.0,
                    1.0,
                    0.5,
                    step=0.01,
                )

                marketing_click_rate = st.slider(
                    "Marketing Click Rate",
                    0.0,
                    1.0,
                    0.25,
                    step=0.01,
                )

                referral_count = st.slider(
                    "Referral Count",
                    0,
                    10,
                    0,
                )

        submitted = st.form_submit_button(
            "Run Churn Analysis",
            type="primary",
            width="stretch",
        )

    if not submitted:
        return None

    return {
        "age": age,
        "gender": gender,
        "country": country,
        "customer_segment": customer_segment,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "monthly_fee": monthly_fee,
        "total_revenue": total_revenue,
        "monthly_logins": monthly_logins,
        "weekly_active_days": weekly_active_days,
        "avg_session_time": avg_session_time,
        "support_tickets": support_tickets,
        "escalations": escalations,
        "avg_resolution_time": avg_resolution_time,
        "csat_score": csat_score,
        "payment_method": payment_method,
        "complaint_type": complaint_type,
        "payment_failures": payment_failures,
        "last_login_days_ago": last_login_days_ago,
        "usage_growth_rate": usage_growth_rate,
        "nps_score": nps_score,
        "email_open_rate": email_open_rate,
        "marketing_click_rate": marketing_click_rate,
        "features_used": features_used,
        "referral_count": referral_count,
        "signup_channel": signup_channel,
        "discount_applied": discount_applied,
        "price_increase_last_3m": price_increase_last_3m,
        "survey_response": survey_response,
    }


def render_prediction(result: dict[str, Any]) -> None:
    st.subheader("Risk Assessment")

    probability = result["churn_probability"]
    risk_level = result["risk_level"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            format_probability(probability),
        )

    with col2:
        st.metric(
            "Risk Classification",
            risk_level,
        )

    with col3:
        st.metric(
            "Intervention Threshold",
            "15%",
        )

    if risk_level == "High":
        st.error(
            "This customer exceeds the retention intervention threshold."
        )

        st.write(
            "The customer is eligible for counterfactual "
            "retention analysis."
        )
    else:
        st.success(
            "This customer is currently below the intervention threshold."
        )

    st.caption(
        "XGBoost produces the probability. The application "
        "then applies a separate 15% intervention threshold."
    )


def render_interventions(
    interventions: list[dict[str, Any]],
) -> None:
    st.subheader("Counterfactual Options")

    st.write(
        "DiCE generates alternative customer profiles that "
        "the churn model predicts would move below the intervention threshold."
    )

    st.info(
        "These are model-based counterfactuals. They do not prove "
        "that making these changes will causally prevent churn."
    )

    if not interventions:
        st.warning("No counterfactual alternatives were generated.")
        return

    for option in interventions:
        option_id = option.get("counterfactual_id")
        probability = option.get("predicted_churn_proba")

        with st.expander(
            (
                f"Option {option_id} - "
                f"Predicted risk {format_probability(probability)}"
            ),
            expanded=True,
        ):
            rows = []

            for change in option.get("interventions", []):
                impact = float(change.get("impact", 0))

                rows.append(
                    {
                        "Feature": format_feature_name(
                            change["feature"]
                        ),
                        "Current": format_value(
                            change["original"]
                        ),
                        "Suggested": format_value(
                            change["suggested"]
                        ),
                        "Model Impact": f"{impact:.2f} pp",
                    }
                )

            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True,
                    width="stretch",
                )


def render_agent_candidates(
    prediction_result: dict[str, Any],
) -> None:
    interventions = prediction_result.get(
        "interventions",
        [],
    )

    st.markdown("#### Candidate Strategies")

    st.write(
        "These are the alternatives the LangGraph agent receives "
        "before making any operational decision."
    )

    if not interventions:
        st.info("No DiCE alternatives are available.")
        return

    for start in range(0, len(interventions), 3):
        options = interventions[start:start + 3]
        columns = st.columns(len(options))

        for column, option in zip(columns, options):
            with column:
                option_id = option.get("counterfactual_id")
                probability = option.get("predicted_churn_proba")

                st.markdown(f"### Option {option_id}")

                st.metric(
                    "Predicted Risk",
                    format_probability(probability),
                )

                for change in option.get("interventions", []):
                    feature = format_feature_name(
                        change["feature"]
                    )

                    original = format_value(
                        change["original"]
                    )

                    suggested = format_value(
                        change["suggested"]
                    )

                    st.write(f"**{feature}**")
                    st.caption(
                        f"{original} -> {suggested}"
                    )


def request_retention_plan_live() -> None:
    payload = {
        "case_id": st.session_state.case_id,
        "customer_profile": st.session_state.customer_data,
        "prediction_payload": st.session_state.prediction_result,
    }

    st.session_state.agent_events = []
    st.session_state.agent_result = None

    current_activity = st.empty()

    execution_status = st.status(
        "Starting LangGraph retention workflow...",
        expanded=True,
    )

    decision_number = 0

    try:
        for event in stream_agent(payload):
            st.session_state.agent_events.append(event)

            event_type = event.get("type")

            if event_type == "run_started":
                execution_status.write(
                    "Retention graph initialized."
                )

            elif event_type == "node_started":
                node_name = event.get("node", "unknown")
                node_info = get_node_info(node_name)

                execution_status.markdown(
                    f"**Node: {node_info['name']}**"
                )

                execution_status.caption(
                    node_info["purpose"]
                )

                if event.get("message"):
                    execution_status.write(
                        event["message"]
                    )

                with current_activity.container():
                    st.info(
                        f"Current graph node: {node_info['name']}"
                    )

                    st.caption(
                        node_info["purpose"]
                    )

            elif event_type == "node_completed":
                node_name = event.get("node", "unknown")
                node_info = get_node_info(node_name)

                execution_status.write(
                    f"Completed: {node_info['name']}"
                )

                data = event.get("data", {})

                if data:
                    execution_status.code(
                        json.dumps(
                            data,
                            indent=2,
                            default=str,
                        )
                    )

            elif event_type == "decision":
                decision_number += 1

                tool_name = event.get("tool", "unknown")
                tool_info = get_tool_info(tool_name)

                decision_reason = event.get(
                    "decision_reason"
                )

                arguments = event.get(
                    "arguments",
                    {},
                )

                execution_status.markdown(
                    (
                        f"**Agent Decision {decision_number}: "
                        f"{tool_info['name']}**"
                    )
                )

                if decision_reason:
                    execution_status.write(
                        decision_reason
                    )

                execution_status.caption(
                    f"Tool purpose: {tool_info['purpose']}"
                )

                if arguments:
                    execution_status.write(
                        "Inputs selected by the agent:"
                    )

                    execution_status.code(
                        json.dumps(
                            arguments,
                            indent=2,
                            default=str,
                        )
                    )

                with current_activity.container():
                    st.warning(
                        f"Agent selected: {tool_info['name']}"
                    )

                    if decision_reason:
                        st.write(
                            decision_reason
                        )

                    st.caption(
                        tool_info["purpose"]
                    )

            elif event_type == "tool_running":
                tool_name = event.get("tool", "unknown")
                tool_info = get_tool_info(tool_name)

                execution_status.markdown(
                    f"**Running Tool: {tool_info['name']}**"
                )

                if event.get("message"):
                    execution_status.write(
                        event["message"]
                    )

                data = event.get("data", {})

                if data:
                    execution_status.code(
                        json.dumps(
                            data,
                            indent=2,
                            default=str,
                        )
                    )

                with current_activity.container():
                    st.info(
                        f"Tool currently running: {tool_info['name']}"
                    )

                    st.caption(
                        tool_info["purpose"]
                    )

            elif event_type == "tool_completed":
                tool_name = event.get("tool", "unknown")
                tool_info = get_tool_info(tool_name)

                execution_status.markdown(
                    f"**Completed Tool: {tool_info['name']}**"
                )

                if event.get("message"):
                    execution_status.write(
                        event["message"]
                    )

                data = event.get("data", {})

                if data:
                    execution_status.write(
                        "Tool result:"
                    )

                    execution_status.code(
                        json.dumps(
                            data,
                            indent=2,
                            default=str,
                        )
                    )

            elif event_type == "node_update":
                node_name = event.get("node", "unknown")
                node_info = get_node_info(node_name)

                message = (
                    f"Graph state updated by {node_info['name']}."
                )

                if event.get("status"):
                    message += (
                        f" State status: {event['status']}"
                    )

                execution_status.caption(
                    message
                )

            elif event_type == "final_plan":
                case_id = event.get("case_id")

                st.session_state.case_id = case_id

                st.session_state.agent_result = {
                    "case_id": case_id,
                    "plan": event.get("plan"),
                    "llm_calls": event.get(
                        "llm_calls",
                        0,
                    ),
                    "events": list(
                        st.session_state.agent_events
                    ),
                }

                current_activity.empty()

                execution_status.update(
                    label="LangGraph retention workflow complete",
                    state="complete",
                    expanded=False,
                )

            elif event_type == "error":
                execution_status.update(
                    label="LangGraph retention workflow failed",
                    state="error",
                    expanded=True,
                )

                execution_status.error(
                    event.get(
                        "message",
                        "Unknown graph error.",
                    )
                )

    except requests.exceptions.ConnectionError:
        execution_status.update(
            label="Could not connect to retention API",
            state="error",
        )

        execution_status.error(
            "Make sure FastAPI is running."
        )

    except requests.exceptions.Timeout:
        execution_status.update(
            label="Retention graph request timed out",
            state="error",
        )

    except RuntimeError as exc:
        execution_status.update(
            label="Retention graph request failed",
            state="error",
        )

        execution_status.error(str(exc))


def render_agent_execution_trace(
    events: list[dict[str, Any]],
) -> None:
    st.markdown("#### LangGraph Execution Trace")

    st.caption(
        "This trace shows observable graph transitions, tool selections, "
        "tool results, and short decision explanations. "
        "It does not expose hidden model chain-of-thought."
    )

    if not events:
        st.info("No execution events were recorded.")
        return

    step = 0

    for event in events:
        event_type = event.get("type")

        if event_type == "node_started":
            step += 1

            node_info = get_node_info(
                event.get("node", "unknown")
            )

            with st.expander(
                f"Step {step}: {node_info['name']}"
            ):
                st.markdown("**Node purpose**")
                st.write(
                    node_info["purpose"]
                )

                if event.get("message"):
                    st.markdown("**Execution**")
                    st.write(
                        event["message"]
                    )

        elif event_type == "decision":
            step += 1

            tool_info = get_tool_info(
                event.get("tool", "unknown")
            )

            with st.expander(
                (
                    f"Step {step}: Agent selected "
                    f"{tool_info['name']}"
                )
            ):
                reason = event.get(
                    "decision_reason"
                )

                if reason:
                    st.markdown(
                        "**Agent explanation**"
                    )

                    st.write(reason)

                st.markdown(
                    "**Tool purpose**"
                )

                st.write(
                    tool_info["purpose"]
                )

                arguments = event.get(
                    "arguments",
                    {},
                )

                if arguments:
                    st.markdown(
                        "**Inputs selected by agent**"
                    )

                    st.json(arguments)

        elif event_type == "tool_completed":
            step += 1

            tool_info = get_tool_info(
                event.get("tool", "unknown")
            )

            with st.expander(
                (
                    f"Step {step}: "
                    f"{tool_info['name']} Result"
                )
            ):
                if event.get("message"):
                    st.write(
                        event["message"]
                    )

                data = event.get(
                    "data",
                    {},
                )

                if data:
                    st.json(data)


def render_agent_result(
    result: dict[str, Any],
) -> None:
    plan = result.get("plan")

    if not plan:
        st.warning(
            "The graph did not return a retention plan."
        )
        return

    st.markdown("### Final Retention Decision")

    selected_option = plan.get(
        "selected_counterfactual_id"
    )

    probability_after = plan.get(
        "expected_churn_probability_after"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Selected Strategy",
            (
                f"DiCE Option {selected_option}"
                if selected_option is not None
                else "Manual / Custom"
            ),
        )

    with col2:
        st.metric(
            "Expected Model Risk",
            format_probability(
                probability_after
            ),
        )

    with col3:
        st.metric(
            "Human Approval",
            (
                "Required"
                if plan.get(
                    "requires_human_approval"
                )
                else "Not Required"
            ),
        )

    with col4:
        st.metric(
            "LLM Calls",
            result.get(
                "llm_calls",
                0,
            ),
        )

    st.markdown("#### Strategy")

    st.write(
        plan.get(
            "strategy_summary",
            "",
        )
    )

    st.markdown(
        "#### Why This Strategy Was Selected"
    )

    st.info(
        plan.get(
            "why_selected",
            "",
        )
    )

    assessments = plan.get(
        "option_assessments",
        [],
    )

    if assessments:
        st.markdown(
            "#### Evaluation of Every DiCE Option"
        )

        rows = []

        for assessment in assessments:
            probability = assessment.get(
                "predicted_churn_probability"
            )

            rows.append(
                {
                    "Option": (
                        f"Option "
                        f"{assessment.get('counterfactual_id')}"
                    ),
                    "Decision": format_feature_name(
                        assessment.get(
                            "decision",
                            "",
                        )
                    ),
                    "Feasibility": format_feature_name(
                        assessment.get(
                            "feasibility",
                            "",
                        )
                    ),
                    "Model Risk": format_probability(
                        probability
                    ),
                    "Summary": assessment.get(
                        "summary",
                        "",
                    ),
                    "Reason": assessment.get(
                        "reason",
                        "",
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
        )

    st.markdown("#### Operational Actions")

    actions = plan.get(
        "actions",
        [],
    )

    if actions:
        rows = []

        for action in actions:
            rows.append(
                {
                    "Action": format_feature_name(
                        action.get(
                            "action_type",
                            "",
                        )
                    ),
                    "Description": action.get(
                        "description",
                        "",
                    ),
                    "Status": format_feature_name(
                        action.get(
                            "status",
                            "",
                        )
                    ),
                    "Approval": (
                        "Required"
                        if action.get(
                            "requires_human_approval"
                        )
                        else "Not Required"
                    ),
                    "Reference": action.get(
                        "tool_reference"
                    )
                    or "",
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
        )

    else:
        st.info(
            "The graph did not create any operational actions."
        )

    events = result.get(
        "events",
        [],
    )

    if events:
        st.divider()

        render_agent_execution_trace(
            events
        )

    limitations = plan.get(
        "limitations",
        [],
    )

    if limitations:
        st.divider()

        with st.expander(
            "Model and Recommendation Limitations"
        ):
            for limitation in limitations:
                st.write(
                    f"- {limitation}"
                )

    case_id = result.get(
        "case_id"
    )

    if case_id:
        st.caption(
            f"Retention case ID: {case_id}"
        )


def render_agent_explanation() -> None:
    with st.expander(
        "How the LangGraph Agent Works"
    ):
        st.graphviz_chart(
            """
            digraph {
                rankdir=LR;

                ML [label="XGBoost + DiCE"];
                State [label="LangGraph State"];
                Init [label="initialize_case"];
                Agent [label="agent"];
                Tools [label="tools"];
                Final [label="finalize_plan"];
                End [label="END"];

                ML -> State;
                State -> Init;
                Init -> Agent;

                Agent -> Tools [label="tool requested"];
                Tools -> Agent [label="tool result"];

                Agent -> Final [label="no more tools"];
                Final -> End;
            }
            """,
            width="stretch",
        )

        st.markdown(
            """
**State**

Contains the customer profile, XGBoost and DiCE results,
message history, execution status, and final plan.

**Agent Node**

The LLM reads the current message state and chooses the next tool.

**Conditional Edge**

Checks whether the LLM produced a tool call.

**Tool Node**

Executes the requested tool and adds the result back into the
message history.

**Agent Loop**

After the tool finishes, execution returns to the Agent Node.
The LLM can change its strategy based on the new information.

**Finalize Node**

Runs when the agent stops requesting tools and converts the
completed graph state into a structured retention plan.
"""
        )


def main() -> None:
    initialize_session_state()

    render_sidebar()

    st.title("Customer Retention Intelligence")

    st.write(
        "Predict churn risk, inspect counterfactual alternatives, "
        "and watch an explicit LangGraph agent determine which "
        "retention strategy is operationally feasible."
    )

    st.divider()

    customer_data = build_customer_form()

    if customer_data is not None:
        st.session_state.customer_data = customer_data
        st.session_state.prediction_result = None
        st.session_state.agent_result = None
        st.session_state.agent_events = []
        st.session_state.case_id = None

        try:
            with st.spinner(
                "Running churn prediction and counterfactual analysis..."
            ):
                prediction_result = call_api(
                    PREDICT_URL,
                    customer_data,
                )

            st.session_state.prediction_result = prediction_result

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the prediction API. "
                "Make sure FastAPI is running."
            )

        except requests.exceptions.Timeout:
            st.error(
                "Prediction request timed out."
            )

        except RuntimeError as exc:
            st.error(str(exc))

    prediction_result = st.session_state.prediction_result

    if prediction_result is None:
        return

    st.divider()

    prediction_tab, dice_tab, agent_tab = st.tabs(
        [
            "Prediction",
            "Counterfactuals",
            "Retention Agent",
        ]
    )

    with prediction_tab:
        render_prediction(
            prediction_result
        )

        with st.expander(
            "Raw Prediction Response"
        ):
            st.json(
                prediction_result
            )

    with dice_tab:
        interventions = prediction_result.get(
            "interventions",
            [],
        )

        if interventions:
            render_interventions(
                interventions
            )

        elif prediction_result.get(
            "risk_level"
        ) == "High":
            st.warning(
                "The customer is high risk, but DiCE did not "
                "generate an actionable counterfactual."
            )

        else:
            st.info(
                "Counterfactuals are only generated for high-risk customers."
            )

    with agent_tab:
        risk_level = prediction_result.get(
            "risk_level"
        )

        st.markdown(
            "### LangGraph Retention Agent"
        )

        render_agent_explanation()

        if risk_level != "High":
            st.info(
                "This customer is below the intervention threshold, "
                "so the retention agent is not activated."
            )
            return

        st.write(
            "The graph receives the customer profile, XGBoost prediction, "
            "and every DiCE alternative. The LLM can then select tools "
            "dynamically and reconsider the case after each tool result."
        )

        st.divider()

        render_agent_candidates(
            prediction_result
        )

        st.divider()

        st.markdown(
            "#### Live LangGraph Execution"
        )

        st.write(
            "Run the agent to watch node transitions, model-selected tools, "
            "business checks, and graph state updates."
        )

        if st.button(
            "Run Retention Agent",
            type="primary",
            width="stretch",
        ):
            request_retention_plan_live()

        if st.session_state.agent_result is not None:
            st.divider()

            render_agent_result(
                st.session_state.agent_result
            )


if __name__ == "__main__":
    main()