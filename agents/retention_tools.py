from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from langchain.tools import ToolRuntime, tool


ACTION_LOG_PATH = "data/agent_actions.jsonl"

DEMO_DISCOUNT_LIMITS = {
    "Enterprise": 10.0,
    "SME": 15.0,
    "Individual": 20.0,
}


def _to_json(data: dict | list) -> str:
    return json.dumps(data, indent=2, default=str)


def _stream(
    runtime: ToolRuntime,
    event: str,
    tool_name: str,
    message: str,
    data: dict | None = None,
) -> None:
    runtime.stream_writer(
        {
            "event": event,
            "tool": tool_name,
            "message": message,
            "data": data or {},
        }
    )


def _load_case_actions(case_id: str) -> list[dict]:
    if not os.path.exists(ACTION_LOG_PATH):
        return []

    actions = []

    with open(ACTION_LOG_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("case_id") == case_id:
                actions.append(record)

    return actions


def _write_action(
    case_id: str,
    action_type: str,
    status: str,
    payload: dict,
) -> dict:
    os.makedirs(os.path.dirname(ACTION_LOG_PATH), exist_ok=True)

    record = {
        "id": str(uuid4()),
        "case_id": case_id,
        "action_type": action_type,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }

    with open(ACTION_LOG_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return record


@tool
def get_customer_context(
    decision_reason: str,
    runtime: ToolRuntime,
) -> str:
    """
    Inspect the customer's account, usage, support, and satisfaction data.

    decision_reason should briefly explain why the agent needs this
    information at the current stage of the retention decision.
    """

    _stream(
        runtime,
        event="tool_running",
        tool_name="get_customer_context",
        message="Loading customer information from LangGraph state.",
    )

    profile = runtime.state["customer_profile"]

    summary = {
        "customer_segment": profile.get("customer_segment"),
        "contract_type": profile.get("contract_type"),
        "monthly_fee": profile.get("monthly_fee"),
        "tenure_months": profile.get("tenure_months"),
        "csat_score": profile.get("csat_score"),
        "support_tickets": profile.get("support_tickets"),
        "escalations": profile.get("escalations"),
        "monthly_logins": profile.get("monthly_logins"),
        "weekly_active_days": profile.get("weekly_active_days"),
    }

    _stream(
        runtime,
        event="tool_completed",
        tool_name="get_customer_context",
        message="Customer information loaded.",
        data=summary,
    )

    return _to_json(profile)


@tool
def get_churn_analysis(
    decision_reason: str,
    runtime: ToolRuntime,
) -> str:
    """
    Inspect the XGBoost churn prediction and every DiCE counterfactual
    generated for the current customer.

    The agent should compare all options instead of automatically
    selecting the first counterfactual.
    """

    _stream(
        runtime,
        event="tool_running",
        tool_name="get_churn_analysis",
        message="Loading churn prediction and DiCE alternatives.",
    )

    payload = runtime.state["prediction_payload"]
    options = payload.get("interventions", [])

    result = {
        "risk_level": payload.get("risk_level"),
        "churn_probability": payload.get("churn_probability"),
        "prediction_class": payload.get("prediction_class"),
        "counterfactual_options": options,
    }

    _stream(
        runtime,
        event="tool_completed",
        tool_name="get_churn_analysis",
        message=f"Loaded {len(options)} counterfactual alternatives.",
        data={
            "risk_level": payload.get("risk_level"),
            "current_probability": payload.get("churn_probability"),
            "number_of_options": len(options),
        },
    )

    return _to_json(result)


@tool
def get_case_actions(
    decision_reason: str,
    runtime: ToolRuntime,
) -> str:
    """
    Check actions already created for this retention case.

    Use this when necessary to avoid creating duplicate tickets or drafts.
    """

    case_id = runtime.state["case_id"]

    _stream(
        runtime,
        event="tool_running",
        tool_name="get_case_actions",
        message="Checking previously created actions for this case.",
    )

    actions = _load_case_actions(case_id)

    _stream(
        runtime,
        event="tool_completed",
        tool_name="get_case_actions",
        message=f"Found {len(actions)} existing actions.",
        data={"action_count": len(actions)},
    )

    return _to_json(actions)


@tool
def evaluate_fee_offer(
    target_monthly_fee: float,
    duration_months: int,
    decision_reason: str,
    runtime: ToolRuntime,
) -> str:
    """
    Check whether a proposed fee reduction follows the demo retention
    policy and estimate its cost.

    Use this before treating a fee-based DiCE option as feasible.
    """

    profile = runtime.state["customer_profile"]

    current_fee = float(profile.get("monthly_fee", 0))
    segment = profile.get("customer_segment", "Individual")
    max_discount = DEMO_DISCOUNT_LIMITS.get(segment, 10.0)

    _stream(
        runtime,
        event="tool_running",
        tool_name="evaluate_fee_offer",
        message="Checking proposed fee against retention policy.",
        data={
            "current_monthly_fee": current_fee,
            "target_monthly_fee": target_monthly_fee,
            "duration_months": duration_months,
        },
    )

    if current_fee <= 0:
        result = {
            "eligible": False,
            "reason": "Current monthly fee is invalid.",
        }

    elif target_monthly_fee <= 0:
        result = {
            "eligible": False,
            "reason": "Target monthly fee must be positive.",
        }

    elif duration_months <= 0:
        result = {
            "eligible": False,
            "reason": "Duration must be at least one month.",
        }

    else:
        discount_percentage = (current_fee - target_monthly_fee) / current_fee * 100
        estimated_cost = max(current_fee - target_monthly_fee, 0) * duration_months

        eligible = 0 < discount_percentage <= max_discount

        if target_monthly_fee >= current_fee:
            reason = "The proposal does not reduce the monthly fee."
        elif eligible:
            reason = "The proposed discount is within the retention policy."
        else:
            reason = "The proposed discount exceeds the maximum allowed discount."

        result = {
            "eligible": eligible,
            "customer_segment": segment,
            "current_monthly_fee": round(current_fee, 2),
            "target_monthly_fee": round(target_monthly_fee, 2),
            "discount_percentage": round(discount_percentage, 2),
            "maximum_allowed_discount": max_discount,
            "duration_months": duration_months,
            "estimated_cost": round(estimated_cost, 2),
            "reason": reason,
        }

    _stream(
        runtime,
        event="tool_completed",
        tool_name="evaluate_fee_offer",
        message="Fee policy evaluation completed.",
        data=result,
    )

    return _to_json(result)


@tool
def check_contract_change(
    target_contract: Literal["Monthly", "Quarterly", "Yearly"],
    decision_reason: str,
    runtime: ToolRuntime,
) -> str:
    """
    Check whether a proposed contract transition is valid.

    Use this before treating a contract-based counterfactual as
    operationally feasible.
    """

    profile = runtime.state["customer_profile"]
    current_contract = profile.get("contract_type")

    _stream(
        runtime,
        event="tool_running",
        tool_name="check_contract_change",
        message="Checking proposed contract transition.",
        data={
            "current_contract": current_contract,
            "target_contract": target_contract,
        },
    )

    contract_rank = {
        "Monthly": 1,
        "Quarterly": 2,
        "Yearly": 3,
    }

    if current_contract not in contract_rank:
        result = {
            "valid": False,
            "reason": "Current contract type is unknown.",
        }

    elif target_contract == current_contract:
        result = {
            "valid": False,
            "reason": "Customer is already on this contract.",
        }

    else:
        result = {
            "valid": True,
            "current_contract": current_contract,
            "target_contract": target_contract,
            "increases_commitment": (
                contract_rank[target_contract] > contract_rank[current_contract]
            ),
            "reason": "Contract transition is valid.",
        }

    _stream(
        runtime,
        event="tool_completed",
        tool_name="check_contract_change",
        message="Contract check completed.",
        data=result,
    )

    return _to_json(result)


@tool
def save_email_draft(
    subject: str,
    body: str,
    decision_reason: str,
    runtime: ToolRuntime,
) -> str:
    """
    Prepare a customer retention email.

    This tool only creates a draft. It never sends the email.
    Customer-facing communication requires human approval.
    """

    case_id = runtime.state["case_id"]

    _stream(
        runtime,
        event="tool_running",
        tool_name="save_email_draft",
        message="Preparing customer retention email.",
    )

    record = _write_action(
        case_id=case_id,
        action_type="email_draft",
        status="prepared",
        payload={
            "subject": subject,
            "body": body,
            "requires_human_approval": True,
        },
    )

    result = {
        "draft_id": record["id"],
        "status": "prepared",
        "requires_human_approval": True,
        "message": "Email draft prepared. It has not been sent.",
    }

    _stream(
        runtime,
        event="tool_completed",
        tool_name="save_email_draft",
        message="Email draft prepared.",
        data={
            "draft_id": record["id"],
            "status": "prepared",
        },
    )

    return _to_json(result)


@tool
def create_csm_ticket(
    priority: Literal["low", "medium", "high", "critical"],
    summary: str,
    reason: str,
    action_steps: list[str],
    decision_reason: str,
    runtime: ToolRuntime,
) -> str:
    """
    Create an internal Customer Success ticket.

    Use this when support, satisfaction, escalation, or account
    management intervention is justified.
    """

    case_id = runtime.state["case_id"]

    _stream(
        runtime,
        event="tool_running",
        tool_name="create_csm_ticket",
        message="Creating Customer Success ticket.",
        data={
            "priority": priority,
            "summary": summary,
        },
    )

    existing_actions = _load_case_actions(case_id)

    for action in existing_actions:
        same_ticket = (
            action.get("action_type") == "csm_ticket"
            and action.get("payload", {}).get("summary") == summary
        )

        if same_ticket:
            result = {
                "ticket_id": action["id"],
                "status": "already_exists",
            }

            _stream(
                runtime,
                event="tool_completed",
                tool_name="create_csm_ticket",
                message="Matching Customer Success ticket already exists.",
                data=result,
            )

            return _to_json(result)

    record = _write_action(
        case_id=case_id,
        action_type="csm_ticket",
        status="created",
        payload={
            "priority": priority,
            "summary": summary,
            "reason": reason,
            "action_steps": action_steps,
        },
    )

    result = {
        "ticket_id": record["id"],
        "status": "created",
    }

    _stream(
        runtime,
        event="tool_completed",
        tool_name="create_csm_ticket",
        message="Customer Success ticket created.",
        data=result,
    )

    return _to_json(result)


RETENTION_TOOLS = [
    get_customer_context,
    get_churn_analysis,
    get_case_actions,
    evaluate_fee_offer,
    check_contract_change,
    save_email_draft,
    create_csm_ticket,
]