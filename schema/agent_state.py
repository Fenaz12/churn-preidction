import operator
from typing import Annotated, Any

from langchain.messages import AnyMessage
from typing_extensions import TypedDict

from schema.retention_models import RetentionPlan


class RetentionState(TypedDict, total=False):
    # Conversation history between the LLM and tools.
    # operator.add means new messages are appended instead of replacing old ones.
    messages: Annotated[list[AnyMessage], operator.add]

    # Unique retention case.
    case_id: str

    # Raw customer values from the API.
    customer_profile: dict[str, Any]

    # XGBoost prediction + DiCE alternatives.
    prediction_payload: dict[str, Any]

    # Useful for UI/debugging.
    llm_calls: int
    status: str

    # Created by the finalization node.
    final_plan: RetentionPlan | None