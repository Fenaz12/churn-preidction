from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

from agents.retention_graph import get_retention_orchestrator

router = APIRouter()

# Instantiate the compiled graph once when the app starts
retention_agent = get_retention_orchestrator()

# Define Pydantic models for the incoming request
class AgentRequest(BaseModel):
    customer_profile: Dict[str, Any]
    prediction_payload: Dict[str, Any]

class ActionPlanResponse(BaseModel):
    parsed_interventions: List[Dict[str, Any]]
    action_plan: Dict[str, str]

@router.post("/orchestrate", response_model=ActionPlanResponse)
async def run_retention_agent(request: AgentRequest):
    initial_state = {
        "customer_profile": request.customer_profile,
        "prediction_payload": request.prediction_payload,
        "parsed_interventions": [],
        "action_plan": {}
    }

    try:
        final_state = await retention_agent.ainvoke(initial_state)
        
        return {
            "parsed_interventions": final_state.get("parsed_interventions", []),
            "action_plan": final_state.get("action_plan", {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Execution Error: {str(e)}")