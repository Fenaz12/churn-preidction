from typing import TypedDict, List, Dict, Any, Annotated

def merge_action_plans(plan1: dict, plan2: dict) -> dict:
    """
    Reducer function to safely merge dictionaries when nodes run in parallel.
    """
    if plan1 is None:
        plan1 = {}
    if plan2 is None:
        plan2 = {}
        
    merged = plan1.copy()
    merged.update(plan2)
    return merged

class RetentionAgentState(TypedDict):
    customer_profile: Dict[str, Any]
    prediction_payload: Dict[str, Any]
    parsed_interventions: List[Dict[str, Any]]
    action_plan: Annotated[Dict[str, str], merge_action_plans]