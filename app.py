"""
Bug Triage OpenEnv - FastAPI Application

Serves the OpenEnv API endpoints: /reset, /step, /state, /tasks, /health
Also serves a Gradio-compatible UI for HuggingFace Spaces.
"""
from __future__ import annotations

import os
import sys
import json
import traceback
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ValidationError

sys.path.insert(0, os.path.dirname(__file__))

from models import Action, ActionType, SeverityLevel, TeamName
from envs.bug_triage_env import BugTriageEnvironment

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Bug Triage OpenEnv",
    description="Real-world software bug triage environment for AI agent training",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment registry (task_id -> env instance)
_envs: Dict[str, BugTriageEnvironment] = {}


def _get_env(task_id: str = "easy_triage") -> BugTriageEnvironment:
    if task_id not in _envs:
        _envs[task_id] = BugTriageEnvironment(task_id=task_id)
    return _envs[task_id]


# ─── Request/Response Models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy_triage"

class StepRequest(BaseModel):
    task_id: str = "easy_triage"
    action: Dict[str, Any]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "bug-triage-openenv", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with descriptions."""
    from tasks.scenarios import ALL_TASKS
    tasks_info = []
    for task_id, task_data in ALL_TASKS.items():
        from openenv_config import TASK_DEFS
        task_def = TASK_DEFS.get(task_id, {})
        tasks_info.append({
            "id": task_id,
            "name": task_def.get("name", task_id),
            "difficulty": task_def.get("difficulty", "unknown"),
            "description": task_def.get("description", ""),
            "max_steps": task_def.get("max_steps", 10),
            "success_threshold": task_def.get("success_threshold", 0.75),
        })
    return {"tasks": tasks_info}


@app.post("/reset")
async def reset(req: ResetRequest):
    """Reset the environment for the given task. Returns initial observation."""
    try:
        env = _get_env(req.task_id)
        obs = env.reset()
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(req: StepRequest):
    """
    Execute one step in the environment.
    Returns: {observation, reward, done, info}
    """
    try:
        env = _get_env(req.task_id)

        try:
            action = Action(**req.action)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=f"Invalid action: {e}")

        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {traceback.format_exc()}")


@app.get("/state")
async def state(task_id: str = "easy_triage"):
    """Return full environment state."""
    try:
        env = _get_env(task_id)
        s = env.state()
        return s.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def home():
    """Simple web UI for the environment."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Bug Triage OpenEnv</title>
  <style>
    body { font-family: sans-serif; background: #0f1117; color: #e0e0e0; padding: 40px; }
    h1 { color: #58a6ff; }
    code { background: #161b22; padding: 2px 6px; border-radius: 4px; color: #79c0ff; }
  </style>
</head>
<body>
  <h1>🐛 Bug Triage OpenEnv</h1>
  <p>A real-world software engineering environment for training AI agents on bug triage.</p>
  <p>API docs: <a href="/docs" style="color:#58a6ff">/docs</a></p>
  <p>Health: <a href="/health" style="color:#58a6ff">/health</a></p>
  <p>Tasks: <a href="/tasks" style="color:#58a6ff">/tasks</a></p>
</body>
</html>
"""


# ─── Main entry point (required by openenv validator) ─────────────────────────

def main():
    """Main entry point — required by openenv validator (entry_point: app:main)."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
