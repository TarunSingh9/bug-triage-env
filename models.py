"""
Typed Pydantic models for Bug Triage OpenEnv environment.
Implements the full OpenEnv spec with Observation, Action, and Reward models.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ─── Enums ───────────────────────────────────────────────────────────────────

class SeverityLevel(str, Enum):
    CRITICAL = "critical"       # P0 - service down / data loss / security breach
    HIGH = "high"               # P1 - major feature broken, many users affected
    MEDIUM = "medium"           # P2 - feature degraded, workaround exists
    LOW = "low"                 # P3 - minor issue, cosmetic, edge case
    INFORMATIONAL = "informational"  # P4 - question / enhancement


class TeamName(str, Enum):
    BACKEND = "backend"
    FRONTEND = "frontend"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    DATA = "data"
    MOBILE = "mobile"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    CLASSIFY_SEVERITY = "classify_severity"
    ASSIGN_TEAM = "assign_team"
    REQUEST_INFO = "request_info"
    ADD_LABEL = "add_label"
    RESOLVE = "resolve"
    ESCALATE = "escalate"
    ADD_COMMENT = "add_comment"


# ─── Environment Info ─────────────────────────────────────────────────────────

class EnvironmentInfo(BaseModel):
    os: Optional[str] = None
    browser: Optional[str] = None
    app_version: Optional[str] = None
    region: Optional[str] = None
    service: Optional[str] = None
    user_count_affected: Optional[int] = None
    error_rate: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None


# ─── History Entry ────────────────────────────────────────────────────────────

class HistoryEntry(BaseModel):
    step: int
    action_type: str
    action_detail: Dict[str, Any]
    reward: float
    timestamp: str


# ─── Observation (what the agent sees) ───────────────────────────────────────

class Observation(BaseModel):
    """The full observation returned to the agent at each step."""
    bug_id: str = Field(..., description="Unique identifier for the bug report")
    title: str = Field(..., description="Short title of the bug report")
    description: str = Field(..., description="Full description of the bug")
    reporter: str = Field(..., description="Username of the reporter")
    created_at: str = Field(..., description="ISO timestamp of creation")
    logs: str = Field(default="", description="Relevant log lines, may be empty")
    stack_trace: str = Field(default="", description="Stack trace if applicable")
    environment: EnvironmentInfo = Field(default_factory=EnvironmentInfo)
    history: List[HistoryEntry] = Field(default_factory=list, description="Steps taken so far")
    step_count: int = Field(default=0, description="Number of steps taken")
    max_steps: int = Field(default=10, description="Maximum steps allowed")
    task_id: str = Field(..., description="Which task is active")
    
    # Running state
    current_severity: Optional[SeverityLevel] = None
    current_team: Optional[TeamName] = None
    current_labels: List[str] = Field(default_factory=list)
    comments: List[str] = Field(default_factory=list)
    info_requested: List[str] = Field(default_factory=list)
    info_received: Optional[Dict[str, str]] = None
    is_escalated: bool = False
    is_resolved: bool = False


# ─── Action (what the agent does) ─────────────────────────────────────────────

class Action(BaseModel):
    """The action the agent takes at each step."""
    action_type: ActionType = Field(..., description="Type of action to perform")
    
    # For classify_severity
    severity: Optional[SeverityLevel] = Field(None, description="Severity classification")
    severity_reasoning: Optional[str] = Field(None, description="Why this severity?")
    
    # For assign_team
    team: Optional[TeamName] = Field(None, description="Team to assign to")
    team_reasoning: Optional[str] = Field(None, description="Why this team?")
    
    # For add_label
    labels: Optional[List[str]] = Field(None, description="Labels to apply")
    
    # For add_comment / general
    comment: Optional[str] = Field(None, description="Comment to add to the bug")
    
    # For resolve
    resolution: Optional[str] = Field(None, description="Resolution description")
    resolution_type: Optional[Literal["fixed", "wont_fix", "duplicate", "cannot_reproduce", "by_design"]] = None
    
    # For request_info
    needs_info: Optional[List[str]] = Field(None, description="What info is needed?")
    
    # For escalate
    escalation_reason: Optional[str] = Field(None, description="Why escalating?")
    escalate_to: Optional[str] = Field(None, description="Who/team to escalate to")
    
    # Investigation hints
    investigation_steps: Optional[List[str]] = Field(None, description="Suggested next investigation steps")


# ─── Reward ───────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    severity_accuracy: float = Field(0.0, description="Correctness of severity (0–1)")
    team_accuracy: float = Field(0.0, description="Correct team assignment (0–1)")
    label_quality: float = Field(0.0, description="Label coverage and precision (0–1)")
    info_gathering: float = Field(0.0, description="Appropriate info requests (0–1)")
    resolution_quality: float = Field(0.0, description="Quality of resolution/escalation (0–1)")
    efficiency: float = Field(0.0, description="Completing task in fewer steps (0–1)")
    investigation_quality: float = Field(0.0, description="Quality of investigation steps (0–1)")
    penalties: float = Field(0.0, description="Penalties for bad actions (negative)")


class Reward(BaseModel):
    """Reward signal returned at each step."""
    value: float = Field(..., description="Total reward for this step, range [-1, 1]")
    cumulative: float = Field(0.0, description="Total cumulative reward so far")
    breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    message: str = Field("", description="Human-readable explanation of reward")
    terminal_score: Optional[float] = Field(None, description="Final 0–1 score if episode done")


# ─── Step Response ────────────────────────────────────────────────────────────

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ─── State ────────────────────────────────────────────────────────────────────

class EnvironmentState(BaseModel):
    """Full internal state of the environment."""
    task_id: str
    episode_id: str
    step_count: int
    max_steps: int
    done: bool
    observation: Observation
    cumulative_reward: float
    reward_history: List[float]
    ground_truth: Dict[str, Any]  # Hidden from agent


# ─── Task Definition ──────────────────────────────────────────────────────────

class TaskDefinition(BaseModel):
    id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int
    success_threshold: float
