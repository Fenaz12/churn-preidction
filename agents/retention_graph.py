from typing import List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from schema.agent_state import RetentionAgentState
from services.llm_client import get_llm

# ── Nodes ─────────────────────────────────────────────────────────────────────

def parse_interventions_node(state: RetentionAgentState) -> RetentionAgentState:
    payload = state["prediction_payload"]
    parsed_actions = []
    
    if payload.get("risk_level") == "High" and payload.get("interventions"):
        parsed_actions = payload["interventions"][0].get("interventions", [])
        
    return {"parsed_interventions": parsed_actions, "action_plan": {}}


def draft_marketing_email_node(state: RetentionAgentState) -> RetentionAgentState:
    llm = get_llm(temperature=0.7)
    interventions = state["parsed_interventions"]
    
    marketing_features = ["monthly_fee", "monthly_logins", "features_used", "weekly_active_days"]
    relevant_actions = [a for a in interventions if a["feature"] in marketing_features]
    
    action_text = "\n".join([f"- {a['feature']}: {a['delta']}" for a in relevant_actions])
    prompt = f"Draft a short, friendly engagement email offering help or a personalized discount to address these usage goals: {action_text}"
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Return just the new data; LangGraph merges it automatically
    return {"action_plan": {"email_draft": response.content}}


def draft_csm_ticket_node(state: RetentionAgentState) -> RetentionAgentState:
    llm = get_llm(temperature=0.2) 
    interventions = state["parsed_interventions"]
    profile = state["customer_profile"]
    
    support_features = ["csat_score", "escalations", "avg_resolution_time", "nps_score", "complaint_type"]
    relevant_actions = [a for a in interventions if a["feature"] in support_features]
    
    action_text = "\n".join([f"- {a['feature']}: {a['delta']} (Impact: {a['impact']}%)" for a in relevant_actions])
    
    prompt = f"""
    Write an urgent, concise internal ticket for our Customer Success Managers.
    Customer Segment: {profile.get('customer_segment')}
    
    The AI model indicates this account will churn unless we resolve these specific support metrics:
    {action_text}
    
    Provide a 3-step action plan for the CSM before they call this customer.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Return just the new data; LangGraph merges it automatically
    return {"action_plan": {"csm_ticket": response.content}}

# ── Routing Logic ─────────────────────────────────────────────────────────────

def route_interventions(state: RetentionAgentState) -> List[str]:
    interventions = state["parsed_interventions"]
    if not interventions:
        return [END]

    routes = []
    features = [action["feature"] for action in interventions]
    
    support_issues = {"csat_score", "escalations", "avg_resolution_time", "nps_score", "support_tickets"}
    marketing_issues = {"monthly_fee", "monthly_logins", "features_used", "weekly_active_days"}
    
    if any(f in support_issues for f in features):
        routes.append("draft_csm_ticket")
    if any(f in marketing_issues for f in features):
        routes.append("draft_marketing_email")
        
    return routes if routes else [END]

# ── Graph Compilation ─────────────────────────────────────────────────────────

def get_retention_orchestrator():
    workflow = StateGraph(RetentionAgentState)

    workflow.add_node("parse_interventions", parse_interventions_node)
    workflow.add_node("draft_marketing_email", draft_marketing_email_node)
    workflow.add_node("draft_csm_ticket", draft_csm_ticket_node)

    workflow.set_entry_point("parse_interventions")

    workflow.add_conditional_edges(
        "parse_interventions",
        route_interventions,
        {
            "draft_marketing_email": "draft_marketing_email",
            "draft_csm_ticket": "draft_csm_ticket",
            END: END
        }
    )

    workflow.add_edge("draft_marketing_email", END)
    workflow.add_edge("draft_csm_ticket", END)

    return workflow.compile()