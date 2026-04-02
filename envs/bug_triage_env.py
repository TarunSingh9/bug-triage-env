"""
Bug Triage OpenEnv Environment
Core environment implementing the full OpenEnv spec.
"""
from __future__ import annotations
import uuid
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Observation, Action, Reward, RewardBreakdown,
    StepResponse, EnvironmentState, HistoryEntry,
    EnvironmentInfo, SeverityLevel, TeamName, ActionType
)
from tasks.scenarios import ALL_TASKS
from graders.task_graders import GRADERS


class BugTriageEnvironment:
    """
    Real-world Bug Triage environment for AI agent training.
    
    Implements the OpenEnv spec:
    - reset() -> Observation
    - step(action) -> (Observation, Reward, done, info)
    - state() -> EnvironmentState
    """

    SEVERITY_WEIGHTS = {
        "critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4, "informational": 0.2
    }

    def __init__(self, task_id: str = "easy_triage"):
        if task_id not in ALL_TASKS:
            raise ValueError(f"Unknown task_id: {task_id}. Choose from: {list(ALL_TASKS.keys())}")
        self.task_id = task_id
        self._task_data = ALL_TASKS[task_id]
        self._gt = self._task_data["ground_truth"]
        self._episode_id: Optional[str] = None
        self._obs: Optional[Observation] = None
        self._cumulative_reward: float = 0.0
        self._reward_history: List[float] = []
        self._obs_history: List[Dict[str, Any]] = []
        self._done: bool = False
        self._step_count: int = 0
        self._info_requested: bool = False

    # ─── OpenEnv Core Methods ─────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment and return initial observation."""
        self._episode_id = str(uuid.uuid4())[:8]
        self._cumulative_reward = 0.0
        self._reward_history = []
        self._obs_history = []
        self._done = False
        self._step_count = 0
        self._info_requested = False

        bug = self._task_data["bug"]
        max_steps = self._get_max_steps()

        env_info = EnvironmentInfo(**{
            k: v for k, v in bug["environment"].items()
            if k in EnvironmentInfo.model_fields
        })

        self._obs = Observation(
            bug_id=bug["bug_id"],
            title=bug["title"],
            description=bug["description"],
            reporter=bug["reporter"],
            created_at=bug["created_at"],
            logs=bug.get("logs", ""),
            stack_trace=bug.get("stack_trace", ""),
            environment=env_info,
            history=[],
            step_count=0,
            max_steps=max_steps,
            task_id=self.task_id,
        )
        return self._obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        Returns: (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self._obs is None:
            raise RuntimeError("Must call reset() before step().")

        self._step_count += 1
        reward_value, breakdown, message = self._compute_step_reward(action)
        
        # Apply action to observation state
        self._apply_action(action)
        self._obs.step_count = self._step_count

        # Check termination
        done = self._check_done(action)
        terminal_score = None
        
        if done:
            self._done = True
            # Run final grader
            obs_dict = self._obs.model_dump()
            grade_result = GRADERS[self.task_id](
                self._obs_history,
                obs_dict,
                self._gt
            )
            terminal_score = grade_result["total"]
            # Add terminal reward bonus
            bonus = (terminal_score - 0.5) * 0.5  # bonus/penalty based on final score
            reward_value = round(reward_value + bonus, 4)
            message += f" | Episode complete. Final score: {terminal_score:.3f}"

        self._cumulative_reward = round(self._cumulative_reward + reward_value, 4)
        self._reward_history.append(reward_value)

        reward = Reward(
            value=reward_value,
            cumulative=self._cumulative_reward,
            breakdown=breakdown,
            message=message,
            terminal_score=terminal_score,
        )

        # Record history entry
        history_entry = HistoryEntry(
            step=self._step_count,
            action_type=action.action_type.value,
            action_detail=action.model_dump(exclude_none=True),
            reward=reward_value,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._obs.history.append(history_entry)
        self._obs_history.append({
            "step": self._step_count,
            "action_detail": action.model_dump(exclude_none=True),
            "reward": reward_value,
        })

        info = {
            "episode_id": self._episode_id,
            "step": self._step_count,
            "task_id": self.task_id,
            "cumulative_reward": self._cumulative_reward,
        }
        if terminal_score is not None:
            info["terminal_score"] = terminal_score
            info["grade_breakdown"] = GRADERS[self.task_id](
                self._obs_history, self._obs.model_dump(), self._gt
            )

        return self._obs, reward, done, info

    def state(self) -> EnvironmentState:
        """Return full environment state (includes ground truth for debugging)."""
        if self._obs is None:
            raise RuntimeError("Must call reset() first.")
        return EnvironmentState(
            task_id=self.task_id,
            episode_id=self._episode_id or "",
            step_count=self._step_count,
            max_steps=self._get_max_steps(),
            done=self._done,
            observation=self._obs,
            cumulative_reward=self._cumulative_reward,
            reward_history=self._reward_history,
            ground_truth=self._gt,
        )

    # ─── Internal Reward Computation ─────────────────────────────────────────

    def _compute_step_reward(self, action: Action) -> Tuple[float, RewardBreakdown, str]:
        """Compute reward for a single step action."""
        bd = RewardBreakdown()
        messages = []
        total = 0.0

        if action.action_type == ActionType.CLASSIFY_SEVERITY:
            if action.severity:
                sev_val = action.severity.value
                gt_sev = self._gt["severity"]
                sev_order = ["informational", "low", "medium", "high", "critical"]
                dist = abs(sev_order.index(sev_val) - sev_order.index(gt_sev))
                bd.severity_accuracy = max(0.0, 1.0 - dist * 0.4)
                step_reward = bd.severity_accuracy * 0.3
                total += step_reward
                if dist == 0:
                    messages.append(f"✓ Correct severity: {sev_val}")
                elif dist == 1:
                    messages.append(f"~ Close severity: {sev_val} (expected {gt_sev})")
                else:
                    messages.append(f"✗ Wrong severity: {sev_val} (expected {gt_sev})")
            else:
                bd.penalties -= 0.1
                total -= 0.1
                messages.append("✗ classify_severity action missing severity field")

        elif action.action_type == ActionType.ASSIGN_TEAM:
            if action.team:
                team_val = action.team.value
                gt_team = self._gt["team"]
                secondary = self._gt.get("secondary_team")
                if team_val == gt_team:
                    bd.team_accuracy = 1.0
                    total += 0.25
                    messages.append(f"✓ Correct team: {team_val}")
                elif team_val == secondary:
                    bd.team_accuracy = 0.7
                    total += 0.17
                    messages.append(f"~ Acceptable team (secondary): {team_val}")
                else:
                    bd.team_accuracy = 0.0
                    bd.penalties -= 0.1
                    total -= 0.1
                    messages.append(f"✗ Wrong team: {team_val} (expected {gt_team})")
            else:
                bd.penalties -= 0.05
                total -= 0.05
                messages.append("✗ assign_team action missing team field")

        elif action.action_type == ActionType.REQUEST_INFO:
            if self._gt.get("needs_info"):
                if action.needs_info:
                    needed_text = " ".join(action.needs_info).lower()
                    req_fields = self._gt.get("required_info_fields", [])
                    hits = sum(1 for f in req_fields if any(kw in needed_text for kw in f.split("_")))
                    bd.info_gathering = hits / max(1, len(req_fields))
                    total += bd.info_gathering * 0.2
                    messages.append(f"~ Info requested ({hits}/{len(req_fields)} relevant fields)")
                    # Provide simulated response
                    if not self._info_requested:
                        self._info_requested = True
                        self._inject_info_response()
                else:
                    bd.penalties -= 0.05
                    total -= 0.05
                    messages.append("✗ request_info missing needs_info field")
            else:
                # Penalize unnecessary info requests
                bd.penalties -= 0.15
                total -= 0.15
                messages.append("✗ Unnecessary info request — enough info already available")

        elif action.action_type == ActionType.ADD_LABEL:
            if action.labels:
                req_labels = self._gt.get("required_labels", [])
                f1 = self._label_f1(action.labels, req_labels)
                bd.label_quality = f1
                total += f1 * 0.15
                messages.append(f"~ Labels added: {action.labels} (F1={f1:.2f})")
            else:
                bd.penalties -= 0.05
                messages.append("✗ add_label missing labels field")

        elif action.action_type == ActionType.ESCALATE:
            if self._gt.get("should_escalate"):
                escalation_score = 0.5
                # Bonus for security escalation in hard task
                if self._gt.get("security_issue"):
                    escalate_text = (
                        (action.escalation_reason or "") + 
                        (action.escalate_to or "") + 
                        (action.comment or "")
                    ).lower()
                    if "security" in escalate_text:
                        escalation_score = 1.0
                        messages.append("✓ Correctly escalated to security team")
                    else:
                        messages.append("~ Escalated but missing security identification")
                else:
                    escalation_score = 1.0
                    messages.append("✓ Correct escalation")
                bd.resolution_quality = escalation_score
                total += escalation_score * 0.25
                self._obs.is_escalated = True
            else:
                bd.penalties -= 0.2
                total -= 0.2
                messages.append("✗ Unnecessary escalation — penalized")

        elif action.action_type == ActionType.ADD_COMMENT:
            # Reward informative comments with investigation steps
            if action.comment or action.investigation_steps:
                steps = action.investigation_steps or []
                comment_text = (action.comment or "") + " ".join(steps)
                keywords = self._gt.get("investigation_steps_keywords", [])
                if keywords:
                    hits = sum(1 for kw in keywords if kw in comment_text.lower())
                    inv_q = min(1.0, hits / max(1, len(keywords) * 0.5))
                    bd.investigation_quality = inv_q
                    total += inv_q * 0.1
                    messages.append(f"~ Comment/investigation quality: {inv_q:.2f}")
                else:
                    total += 0.05  # Generic small reward for commenting
                    messages.append("+ Comment added")
                if action.comment:
                    self._obs.comments.append(action.comment)
            else:
                bd.penalties -= 0.05
                messages.append("✗ Empty comment")

        elif action.action_type == ActionType.RESOLVE:
            # Only give credit if we've already classified + assigned
            if self._obs.current_severity and self._obs.current_team:
                if action.resolution and len(action.resolution) > 20:
                    gt_sev = self._gt["severity"]
                    gt_team = self._gt["team"]
                    sev_ok = (self._obs.current_severity.value == gt_sev)
                    team_ok = (self._obs.current_team.value == gt_team)
                    res_score = (0.5 if sev_ok else 0.0) + (0.5 if team_ok else 0.0)
                    bd.resolution_quality = res_score
                    total += res_score * 0.2
                    messages.append(f"~ Resolution (sev_ok={sev_ok}, team_ok={team_ok})")
                    self._obs.is_resolved = True
                else:
                    bd.penalties -= 0.1
                    total -= 0.1
                    messages.append("✗ Resolution too short or missing")
            else:
                bd.penalties -= 0.2
                total -= 0.2
                messages.append("✗ Cannot resolve before classifying severity and assigning team")

        # Step penalty for using too many steps
        max_steps = self._get_max_steps()
        if self._step_count > max_steps * 0.8:
            bd.penalties -= 0.02
            total -= 0.02
            messages.append(f"~ Nearing step limit ({self._step_count}/{max_steps})")

        total = round(max(-1.0, min(1.0, total + bd.penalties)), 4)
        return total, bd, " | ".join(messages) if messages else "No reward signal"

    def _apply_action(self, action: Action) -> None:
        """Apply action side effects to observation state."""
        if action.action_type == ActionType.CLASSIFY_SEVERITY and action.severity:
            self._obs.current_severity = action.severity
        elif action.action_type == ActionType.ASSIGN_TEAM and action.team:
            self._obs.current_team = action.team
        elif action.action_type == ActionType.ADD_LABEL and action.labels:
            for label in action.labels:
                if label not in self._obs.current_labels:
                    self._obs.current_labels.append(label)
        elif action.action_type == ActionType.REQUEST_INFO and action.needs_info:
            self._obs.info_requested.extend(action.needs_info)
        elif action.action_type == ActionType.ESCALATE:
            self._obs.is_escalated = True
        elif action.action_type == ActionType.RESOLVE:
            self._obs.is_resolved = True
        elif action.action_type == ActionType.ADD_COMMENT and action.comment:
            self._obs.comments.append(action.comment)

    def _check_done(self, action: Action) -> bool:
        """Check if the episode should terminate."""
        # Explicit termination actions
        if action.action_type in (ActionType.RESOLVE, ActionType.ESCALATE):
            # For easy task, resolve can end; for medium/hard, escalate ends
            if self.task_id == "easy_triage" and action.action_type == ActionType.RESOLVE:
                return True
            if self.task_id in ("medium_investigation", "hard_triage") and action.action_type == ActionType.ESCALATE:
                return True

        # Max steps
        if self._step_count >= self._get_max_steps():
            return True

        # Task-specific: easy task is done if severity + team + labels all set
        if self.task_id == "easy_triage":
            if (self._obs.current_severity and 
                self._obs.current_team and 
                len(self._obs.current_labels) >= 2):
                return True

        return False

    def _inject_info_response(self) -> None:
        """Inject simulated info response into the observation for medium task."""
        if "info_responses" in self._task_data:
            responses = self._task_data["info_responses"]
            combined = "\n\n--- ADDITIONAL INFORMATION PROVIDED ---\n"
            for key, val in responses.items():
                combined += f"\n[{key}]:\n{val}\n"
            self._obs.logs += combined
            self._obs.info_received = responses

    def _get_max_steps(self) -> int:
        task_max_steps = {
            "easy_triage": 5,
            "medium_investigation": 8,
            "hard_triage": 10,
        }
        return task_max_steps.get(self.task_id, 10)

    @staticmethod
    def _label_f1(predicted: List[str], required: List[str]) -> float:
        if not predicted or not required:
            return 0.0
        pred_set = {l.lower() for l in predicted}
        req_set = set(required)
        hits = sum(1 for req in req_set for pred in pred_set if req in pred or pred in req)
        prec = hits / len(pred_set)
        rec = hits / len(req_set)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)
