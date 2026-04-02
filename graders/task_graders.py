"""
Deterministic graders for all 3 tasks.
Each grader scores agent performance 0.0–1.0.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import re


def _severity_score(predicted: Optional[str], ground_truth: str) -> float:
    """Score severity prediction with partial credit for adjacent levels."""
    if predicted is None:
        return 0.0
    severity_order = ["informational", "low", "medium", "high", "critical"]
    if predicted == ground_truth:
        return 1.0
    try:
        pred_idx = severity_order.index(predicted)
        true_idx = severity_order.index(ground_truth)
        distance = abs(pred_idx - true_idx)
        return max(0.0, 1.0 - distance * 0.4)
    except ValueError:
        return 0.0


def _team_score(predicted: Optional[str], ground_truth: str, secondary: Optional[str] = None) -> float:
    """Score team assignment."""
    if predicted is None:
        return 0.0
    if predicted == ground_truth:
        return 1.0
    if secondary and predicted == secondary:
        return 0.7  # Partial credit for correct secondary team
    return 0.0


def _label_score(predicted_labels: List[str], required_labels: List[str]) -> float:
    """F1-style label scoring."""
    if not predicted_labels:
        return 0.0
    predicted_lower = {l.lower() for l in predicted_labels}
    required_lower = set(required_labels)
    
    # Partial matching: label is "hit" if required label is substring or vice versa
    hits = 0
    for req in required_lower:
        for pred in predicted_lower:
            if req in pred or pred in req:
                hits += 1
                break
    
    precision = hits / len(predicted_lower) if predicted_lower else 0.0
    recall = hits / len(required_lower) if required_lower else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _info_request_score(requested: List[str], required_fields: List[str]) -> float:
    """Score how well the agent requested the right information."""
    if not required_fields:
        return 1.0  # No info needed, no penalty for not asking
    if not requested:
        return 0.0
    requested_text = " ".join(requested).lower()
    hits = sum(1 for field in required_fields if any(
        kw in requested_text for kw in field.lower().split("_")
    ))
    return hits / len(required_fields)


def _investigation_steps_score(steps: List[str], required_keywords: List[str]) -> float:
    """Score quality of investigation steps based on keyword coverage."""
    if not steps:
        return 0.0
    steps_text = " ".join(steps).lower()
    hits = sum(1 for kw in required_keywords if kw in steps_text)
    return min(1.0, hits / max(1, len(required_keywords) * 0.6))  # Need 60% coverage for full score


def grade_easy_task(obs_history: List[Dict[str, Any]], final_obs: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, float]:
    """
    Grade easy triage task.
    Weights: severity(35%), team(35%), labels(20%), efficiency(10%)
    """
    severity_pred = final_obs.get("current_severity")
    team_pred = final_obs.get("current_team")
    labels_pred = final_obs.get("current_labels", [])
    step_count = final_obs.get("step_count", 10)
    max_steps = final_obs.get("max_steps", 5)

    sev_score = _severity_score(severity_pred, gt["severity"])
    team_sc = _team_score(team_pred, gt["team"])
    label_sc = _label_score(labels_pred, gt["required_labels"])
    
    # Efficiency: full credit if done in ≤3 steps, linear decay to 0 at max_steps
    efficiency = max(0.0, 1.0 - max(0, step_count - 3) / (max_steps - 3 + 1))

    total = (
        sev_score * 0.35 +
        team_sc * 0.35 +
        label_sc * 0.20 +
        efficiency * 0.10
    )
    return {
        "total": round(total, 4),
        "severity_accuracy": sev_score,
        "team_accuracy": team_sc,
        "label_quality": label_sc,
        "efficiency": efficiency,
    }


def grade_medium_task(obs_history: List[Dict[str, Any]], final_obs: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, float]:
    """
    Grade medium investigation task.
    Weights: info_gathering(20%), severity(25%), team(20%), escalation(20%), labels(15%)
    """
    severity_pred = final_obs.get("current_severity")
    team_pred = final_obs.get("current_team")
    labels_pred = final_obs.get("current_labels", [])
    info_requested = final_obs.get("info_requested", [])
    is_escalated = final_obs.get("is_escalated", False)
    step_count = final_obs.get("step_count", 10)

    sev_score = _severity_score(severity_pred, gt["severity"])
    team_sc = _team_score(team_pred, gt["team"])
    label_sc = _label_score(labels_pred, gt["required_labels"])
    info_sc = _info_request_score(info_requested, gt.get("required_info_fields", []))
    
    # Escalation: required for this task due to financial impact
    escalation_sc = 1.0 if is_escalated else 0.0
    
    # Efficiency bonus/penalty: medium task expects ~5-6 steps
    efficiency = max(0.0, 1.0 - max(0, step_count - 5) / 5)

    total = (
        info_sc * 0.20 +
        sev_score * 0.25 +
        team_sc * 0.20 +
        escalation_sc * 0.20 +
        label_sc * 0.15
    )
    return {
        "total": round(total, 4),
        "info_gathering": info_sc,
        "severity_accuracy": sev_score,
        "team_accuracy": team_sc,
        "escalation": escalation_sc,
        "label_quality": label_sc,
        "efficiency": efficiency,
    }


def grade_hard_task(obs_history: List[Dict[str, Any]], final_obs: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, float]:
    """
    Grade hard triage task.
    Weights: security_identification(25%), escalation(20%), severity(15%), team(15%), labels(15%), investigation(10%)
    
    Hard task: agent must identify the security (data leakage) implication, escalate to security,
    classify correctly as critical, and provide a meaningful investigation plan.
    """
    severity_pred = final_obs.get("current_severity")
    team_pred = final_obs.get("current_team")
    labels_pred = final_obs.get("current_labels", [])
    is_escalated = final_obs.get("is_escalated", False)
    comments = final_obs.get("comments", [])
    
    # Collect all investigation steps from comments and history
    all_text = " ".join(comments).lower()
    investigation_steps = []
    for entry in obs_history:
        action = entry.get("action_detail", {})
        if action.get("investigation_steps"):
            investigation_steps.extend(action["investigation_steps"])
        if action.get("comment"):
            investigation_steps.append(action["comment"])
    
    # Security identification: did the agent recognize the data exposure?
    security_keywords = ["security", "data leak", "data exposure", "tenant", "cross-tenant", "pii", "isolation"]
    security_identified = any(kw in all_text for kw in security_keywords)
    # Also check if security label was applied
    if any("security" in l.lower() or "data-leak" in l.lower() or "tenant" in l.lower() 
           for l in labels_pred):
        security_identified = True
    security_sc = 1.0 if security_identified else 0.0

    sev_score = _severity_score(severity_pred, gt["severity"])
    team_sc = _team_score(team_pred, gt["team"], gt.get("secondary_team"))
    label_sc = _label_score(labels_pred, gt["required_labels"])
    
    # Escalation with security focus
    escalation_sc = 0.0
    if is_escalated:
        escalation_sc = 0.6  # Base escalation credit
        # Bonus if escalated to security specifically
        comments_text = " ".join(comments + [str(obs_history)]).lower()
        if "security" in comments_text:
            escalation_sc = 1.0
    
    # Investigation quality
    inv_sc = _investigation_steps_score(
        investigation_steps, 
        gt["investigation_steps_keywords"]
    )

    total = (
        security_sc * 0.25 +
        escalation_sc * 0.20 +
        sev_score * 0.15 +
        team_sc * 0.15 +
        label_sc * 0.15 +
        inv_sc * 0.10
    )
    return {
        "total": round(total, 4),
        "security_identification": security_sc,
        "escalation": escalation_sc,
        "severity_accuracy": sev_score,
        "team_accuracy": team_sc,
        "label_quality": label_sc,
        "investigation_quality": inv_sc,
    }


GRADERS = {
    "easy_triage": grade_easy_task,
    "medium_investigation": grade_medium_task,
    "hard_triage": grade_hard_task,
}
