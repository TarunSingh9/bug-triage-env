"""
Unit tests for Bug Triage OpenEnv.
Run: python -m pytest tests/test_env.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from models import Action, ActionType, SeverityLevel, TeamName
from envs.bug_triage_env import BugTriageEnvironment
from graders.task_graders import (
    grade_easy_task, grade_medium_task, grade_hard_task,
    _severity_score, _team_score, _label_score
)


# ─── Grader Unit Tests ────────────────────────────────────────────────────────

class TestGraderUtilities:
    def test_severity_exact_match(self):
        assert _severity_score("high", "high") == 1.0

    def test_severity_adjacent(self):
        score = _severity_score("medium", "high")
        assert 0.5 < score < 1.0

    def test_severity_far_off(self):
        score = _severity_score("informational", "critical")
        assert score < 0.1

    def test_severity_none(self):
        assert _severity_score(None, "high") == 0.0

    def test_team_exact(self):
        assert _team_score("frontend", "frontend") == 1.0

    def test_team_wrong(self):
        assert _team_score("backend", "frontend") == 0.0

    def test_team_secondary(self):
        score = _team_score("security", "infrastructure", "security")
        assert score == 0.7

    def test_label_f1_perfect(self):
        score = _label_score(["safari", "regression", "login"], ["safari", "regression", "login"])
        assert score == 1.0

    def test_label_f1_partial(self):
        score = _label_score(["safari", "regression"], ["safari", "regression", "login", "v2.4.1"])
        assert 0.4 < score < 0.8

    def test_label_f1_empty(self):
        assert _label_score([], ["safari"]) == 0.0


# ─── Environment Tests ────────────────────────────────────────────────────────

class TestEnvironmentReset:
    def test_reset_easy(self):
        env = BugTriageEnvironment("easy_triage")
        obs = env.reset()
        assert obs.bug_id == "BUG-4821"
        assert obs.step_count == 0
        assert obs.current_severity is None
        assert obs.current_team is None
        assert obs.task_id == "easy_triage"

    def test_reset_medium(self):
        env = BugTriageEnvironment("medium_investigation")
        obs = env.reset()
        assert obs.bug_id == "BUG-5503"
        assert obs.step_count == 0

    def test_reset_hard(self):
        env = BugTriageEnvironment("hard_triage")
        obs = env.reset()
        assert obs.bug_id == "BUG-6001"
        assert obs.step_count == 0

    def test_invalid_task(self):
        with pytest.raises(ValueError):
            BugTriageEnvironment("nonexistent_task")

    def test_reset_clears_state(self):
        env = BugTriageEnvironment("easy_triage")
        env.reset()
        action = Action(action_type=ActionType.CLASSIFY_SEVERITY, severity=SeverityLevel.HIGH)
        env.step(action)
        env.reset()
        obs = env.reset()
        assert obs.step_count == 0
        assert obs.current_severity is None


class TestEnvironmentStep:
    def setup_method(self):
        self.env = BugTriageEnvironment("easy_triage")
        self.env.reset()

    def test_step_returns_correct_types(self):
        action = Action(action_type=ActionType.CLASSIFY_SEVERITY, severity=SeverityLevel.HIGH)
        obs, reward, done, info = self.env.step(action)
        assert obs.step_count == 1
        assert isinstance(reward.value, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_correct_severity_gives_positive_reward(self):
        action = Action(action_type=ActionType.CLASSIFY_SEVERITY, severity=SeverityLevel.HIGH)
        _, reward, _, _ = self.env.step(action)
        assert reward.value > 0

    def test_wrong_severity_gives_lower_reward(self):
        env2 = BugTriageEnvironment("easy_triage")
        env2.reset()
        action_wrong = Action(action_type=ActionType.CLASSIFY_SEVERITY, severity=SeverityLevel.LOW)
        _, reward_wrong, _, _ = env2.step(action_wrong)
        
        env3 = BugTriageEnvironment("easy_triage")
        env3.reset()
        action_right = Action(action_type=ActionType.CLASSIFY_SEVERITY, severity=SeverityLevel.HIGH)
        _, reward_right, _, _ = env3.step(action_right)
        
        assert reward_right.value > reward_wrong.value

    def test_step_increments_count(self):
        for i in range(3):
            action = Action(action_type=ActionType.ADD_COMMENT, comment="investigating...")
            obs, _, _, _ = self.env.step(action)
        assert obs.step_count == 3

    def test_step_after_done_raises(self):
        # Use up all steps
        env = BugTriageEnvironment("easy_triage")
        env.reset()
        for _ in range(5):
            action = Action(action_type=ActionType.ADD_COMMENT, comment="test")
            _, _, done, _ = env.step(action)
            if done:
                break
        with pytest.raises(RuntimeError):
            env.step(Action(action_type=ActionType.ADD_COMMENT, comment="after done"))

    def test_step_without_reset_raises(self):
        env = BugTriageEnvironment("easy_triage")
        with pytest.raises(RuntimeError):
            env.step(Action(action_type=ActionType.ADD_COMMENT, comment="test"))

    def test_state_returns_correct_structure(self):
        state = self.env.state()
        assert state.task_id == "easy_triage"
        assert state.step_count == 0
        assert not state.done
        assert "severity" in state.ground_truth


class TestEasyTaskCompletion:
    def test_full_correct_episode(self):
        env = BugTriageEnvironment("easy_triage")
        env.reset()

        actions = [
            Action(action_type=ActionType.CLASSIFY_SEVERITY, severity=SeverityLevel.HIGH,
                   severity_reasoning="Login broken for many Safari users"),
            Action(action_type=ActionType.ASSIGN_TEAM, team=TeamName.FRONTEND,
                   team_reasoning="Safari-specific JS issue in frontend bundle"),
            Action(action_type=ActionType.ADD_LABEL, labels=["safari", "regression", "login", "v2.4.1"]),
        ]

        total_reward = 0
        terminal_score = None
        for action in actions:
            obs, reward, done, info = env.step(action)
            total_reward += reward.value
            if info.get("terminal_score") is not None:
                terminal_score = info["terminal_score"]

        assert terminal_score is not None
        assert terminal_score >= 0.7, f"Easy task should score ≥0.7, got {terminal_score}"
        assert total_reward > 0

    def test_episode_terminates(self):
        env = BugTriageEnvironment("easy_triage")
        env.reset()

        # Run max_steps actions
        for i in range(6):  # More than max_steps
            action = Action(action_type=ActionType.ADD_COMMENT, comment="step")
            _, _, done, _ = env.step(action)
            if done:
                break
        assert done


class TestMediumTaskRewardSignals:
    def test_unnecessary_info_request_penalized(self):
        """Easy task has enough info — requesting info should be penalized."""
        env = BugTriageEnvironment("easy_triage")
        env.reset()
        action = Action(
            action_type=ActionType.REQUEST_INFO,
            needs_info=["more logs please"],
        )
        _, reward, _, _ = env.step(action)
        assert reward.value < 0, "Unnecessary info request should be penalized"

    def test_info_request_rewarded_on_medium(self):
        """Medium task needs info — requesting it should be rewarded."""
        env = BugTriageEnvironment("medium_investigation")
        env.reset()
        action = Action(
            action_type=ActionType.REQUEST_INFO,
            needs_info=["payment_provider_logs", "affected_user_ids", "recent_deploys"],
        )
        _, reward, _, _ = env.step(action)
        assert reward.value > 0, "Correct info request should give positive reward"

    def test_unnecessary_escalation_penalized(self):
        env = BugTriageEnvironment("easy_triage")
        env.reset()
        action = Action(
            action_type=ActionType.ESCALATE,
            escalate_to="CEO",
            escalation_reason="Login broken",
        )
        _, reward, _, _ = env.step(action)
        assert reward.value < 0


class TestHardTaskSecurityDetection:
    def test_security_label_improves_score(self):
        env = BugTriageEnvironment("hard_triage")
        env.reset()

        # Take key correct actions
        actions = [
            Action(action_type=ActionType.CLASSIFY_SEVERITY, severity=SeverityLevel.CRITICAL),
            Action(action_type=ActionType.ADD_LABEL, labels=["security", "data-leak", "tenant-isolation", "critical"]),
            Action(action_type=ActionType.ESCALATE,
                   escalate_to="security-oncall",
                   escalation_reason="Tenant isolation breach — users seeing other tenants data",
                   comment="security data exposure requires immediate security team involvement"),
        ]

        terminal_score = None
        for action in actions:
            obs, reward, done, info = env.step(action)
            if info.get("terminal_score") is not None:
                terminal_score = info["terminal_score"]

        if terminal_score is not None:
            assert terminal_score >= 0.5, f"Hard task with security labels should score ≥0.5, got {terminal_score}"


class TestGraderScoreRange:
    """All graders must return scores in [0.0, 1.0]."""

    def _run_and_grade(self, task_id: str) -> float:
        from graders.task_graders import GRADERS
        env = BugTriageEnvironment(task_id)
        env.reset()
        # Run random plausible actions
        action = Action(action_type=ActionType.ADD_COMMENT, comment="investigating the issue thoroughly")
        obs, _, done, info = env.step(action)
        
        # Force terminate
        for _ in range(20):
            if done:
                break
            obs, _, done, info = env.step(
                Action(action_type=ActionType.ADD_COMMENT, comment="more analysis")
            )
        
        grade = GRADERS[task_id](
            [{"step": 1, "action_detail": {}, "reward": 0.0}],
            obs.model_dump(),
            env._gt,
        )
        return grade["total"]

    def test_easy_score_in_range(self):
        score = self._run_and_grade("easy_triage")
        assert 0.0 <= score <= 1.0

    def test_medium_score_in_range(self):
        score = self._run_and_grade("medium_investigation")
        assert 0.0 <= score <= 1.0

    def test_hard_score_in_range(self):
        score = self._run_and_grade("hard_triage")
        assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
