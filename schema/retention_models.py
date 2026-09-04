from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CounterfactualAssessment(BaseModel):
    counterfactual_id: int
    predicted_churn_probability: Optional[float] = Field(default=None, ge=0, le=1)

    decision: Literal[
        "selected",
        "rejected",
        "considered",
    ]

    feasibility: Literal[
        "feasible",
        "partially_feasible",
        "not_feasible",
        "uncertain",
    ]

    summary: str
    reason: str


class RetentionAction(BaseModel):
    action_type: Literal[
        "fee_offer",
        "contract_change",
        "email_draft",
        "csm_ticket",
        "monitoring",
        "manual_review",
        "no_action",
    ]

    description: str

    status: Literal[
        "proposed",
        "prepared",
        "created",
        "not_needed",
    ]

    tool_reference: Optional[str] = None
    requires_human_approval: bool = True


class RetentionPlan(BaseModel):
    selected_counterfactual_id: Optional[int] = Field(
        default=None,
        description="Selected DiCE option. None if no option is operationally suitable.",
    )

    expected_churn_probability_after: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    strategy_summary: str
    why_selected: str

    option_assessments: List[CounterfactualAssessment]
    actions: List[RetentionAction]

    requires_human_approval: bool
    limitations: List[str]