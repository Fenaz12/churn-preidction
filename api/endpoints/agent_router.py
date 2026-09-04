import json
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.retention_graph import get_retention_graph
from schema.retention_models import RetentionPlan


router = APIRouter()


NODE_PURPOSES = {
    "initialize_case": (
        "Creates the initial shared LangGraph state for the retention case."
    ),
    "agent": (
        "The LLM examines current state and decides which tool should run next."
    ),
    "tools": (
        "Executes model-selected tools and adds their results to graph state."
    ),
    "finalize_plan": (
        "Converts the completed graph state into a structured retention plan."
    ),
}


class AgentRequest(BaseModel):
    case_id: Optional[str] = None
    customer_profile: Dict[str, Any]
    prediction_payload: Dict[str, Any]


class AgentResponse(BaseModel):
    case_id: str
    plan: RetentionPlan
    llm_calls: int


def create_initial_state(case_id: str, request: AgentRequest) -> dict:
    return {
        "messages": [],
        "case_id": case_id,
        "customer_profile": request.customer_profile,
        "prediction_payload": request.prediction_payload,
        "llm_calls": 0,
        "status": "new",
        "final_plan": None,
    }


def stream_event(event_type: str, **payload) -> str:
    return json.dumps(
        {
            "type": event_type,
            **payload,
        },
        default=str,
    ) + "\n"


@router.post("/orchestrate", response_model=AgentResponse)
def run_retention_graph(request: AgentRequest):
    case_id = request.case_id or str(uuid4())

    graph = get_retention_graph()

    config = {
        "configurable": {
            "thread_id": case_id,
        },
        "recursion_limit": 30,
    }

    initial_state = create_initial_state(case_id, request)

    try:
        result = graph.invoke(
            initial_state,
            config=config,
        )

        plan = result.get("final_plan")

        if plan is None:
            raise RuntimeError("Graph finished without a final retention plan.")

        return {
            "case_id": case_id,
            "plan": plan,
            "llm_calls": result.get("llm_calls", 0),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Retention graph error: {str(exc)}",
        ) from exc


@router.post("/orchestrate/stream")
def stream_retention_graph(request: AgentRequest):
    case_id = request.case_id or str(uuid4())

    graph = get_retention_graph()

    config = {
        "configurable": {
            "thread_id": case_id,
        },
        "recursion_limit": 30,
    }

    initial_state = create_initial_state(case_id, request)

    def event_generator():
        yield stream_event(
            "run_started",
            case_id=case_id,
            message="LangGraph retention workflow started.",
        )

        try:
            for chunk in graph.stream(
                initial_state,
                config=config,
                stream_mode=["updates", "custom"],
                version="v2",
            ):
                chunk_type = chunk["type"]

                if chunk_type == "custom":
                    custom = dict(chunk["data"])
                    event_type = custom.pop("event", "progress")

                    yield stream_event(
                        event_type,
                        **custom,
                    )

                elif chunk_type == "updates":
                    updates = chunk["data"]

                    for node_name, update in updates.items():
                        state_status = None

                        if isinstance(update, dict):
                            state_status = update.get("status")

                        yield stream_event(
                            "node_update",
                            node=node_name,
                            purpose=NODE_PURPOSES.get(node_name, ""),
                            status=state_status,
                        )

            snapshot = graph.get_state(config)
            values = snapshot.values

            final_plan = values.get("final_plan")

            if final_plan is None:
                raise RuntimeError(
                    "Graph completed without producing a final retention plan."
                )

            if hasattr(final_plan, "model_dump"):
                plan_data = final_plan.model_dump()
            else:
                plan_data = final_plan

            yield stream_event(
                "final_plan",
                case_id=case_id,
                llm_calls=values.get("llm_calls", 0),
                plan=plan_data,
            )

        except Exception as exc:
            yield stream_event(
                "error",
                case_id=case_id,
                message=str(exc),
            )

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
    )