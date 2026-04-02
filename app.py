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
        
        # Validate action
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
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bug Triage OpenEnv</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; min-height: 100vh; }
  .hero { background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%); padding: 60px 40px; text-align: center; border-bottom: 1px solid #2a2f3e; }
  .hero h1 { font-size: 2.5em; color: #58a6ff; margin-bottom: 12px; }
  .hero p { font-size: 1.1em; color: #8b949e; max-width: 600px; margin: 0 auto; }
  .badge { display: inline-block; background: #238636; color: #fff; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; margin: 8px 4px; }
  .badge.blue { background: #1f6feb; }
  .badge.purple { background: #6e40c9; }
  .content { max-width: 900px; margin: 40px auto; padding: 0 20px; }
  .card { background: #1c2030; border: 1px solid #2a2f3e; border-radius: 12px; padding: 24px; margin-bottom: 24px; }
  .card h2 { color: #58a6ff; margin-bottom: 16px; font-size: 1.2em; }
  .task { background: #161b22; border: 1px solid #2a2f3e; border-radius: 8px; padding: 16px; margin: 10px 0; }
  .task-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .diff { padding: 3px 10px; border-radius: 12px; font-size: 0.75em; font-weight: bold; }
  .easy { background: #238636; color: #fff; }
  .medium { background: #d29922; color: #000; }
  .hard { background: #da3633; color: #fff; }
  code { background: #161b22; padding: 2px 6px; border-radius: 4px; font-family: monospace; color: #79c0ff; }
  pre { background: #161b22; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 0.85em; color: #c9d1d9; line-height: 1.6; }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .endpoint { display: flex; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid #2a2f3e; }
  .method { font-weight: bold; padding: 3px 10px; border-radius: 4px; font-size: 0.8em; min-width: 50px; text-align: center; }
  .post { background: #1a3a1a; color: #3fb950; border: 1px solid #3fb950; }
  .get { background: #1a2a3a; color: #58a6ff; border: 1px solid #58a6ff; }
</style>
</head>
<body>
<div class="hero">
  <h1>🐛 Bug Triage OpenEnv</h1>
  <p>A real-world software engineering environment for training and evaluating AI agents on bug triage tasks.</p>
  <br>
  <span class="badge">OpenEnv Compliant</span>
  <span class="badge blue">3 Tasks</span>
  <span class="badge purple">Real-World Domain</span>
</div>
<div class="content">
  <div class="card">
    <h2>📋 Tasks</h2>
    <div class="task">
      <div class="task-header"><span class="diff easy">EASY</span><strong>Basic Bug Severity Classification</strong></div>
      <p style="color:#8b949e; font-size:0.9em">Clear regression bug — classify severity, assign team, add labels within 5 steps.</p>
    </div>
    <div class="task">
      <div class="task-header"><span class="diff medium">MEDIUM</span><strong>Incomplete Bug Report Investigation</strong></div>
      <p style="color:#8b949e; font-size:0.9em">Incomplete payment bug — request missing info, uncover a duplicate charge, escalate.</p>
    </div>
    <div class="task">
      <div class="task-header"><span class="diff hard">HARD</span><strong>Ambiguous Multi-System Incident</strong></div>
      <p style="color:#8b949e; font-size:0.9em">Multi-system incident with tenant data exposure — identify security implications, escalate to security team.</p>
    </div>
  </div>
  <div class="card">
    <h2>🔌 API Endpoints</h2>
    <div class="endpoint"><span class="method get">GET</span><code>/health</code><span>Health check</span></div>
    <div class="endpoint"><span class="method get">GET</span><code>/tasks</code><span>List all tasks</span></div>
    <div class="endpoint"><span class="method post">POST</span><code>/reset</code><span>Reset environment, get initial observation</span></div>
    <div class="endpoint"><span class="method post">POST</span><code>/step</code><span>Execute an action, get observation + reward</span></div>
    <div class="endpoint"><span class="method get">GET</span><code>/state</code><span>Get full environment state</span></div>
    <div class="endpoint" style="border:none"><span class="method get">GET</span><code>/docs</code><span>Interactive API documentation</span></div>
  </div>
  <div class="card">
    <h2>⚡ Quick Start</h2>
<pre>import requests

BASE = "https://your-space.hf.space"

# 1. Reset environment
obs = requests.post(f"{BASE}/reset", json={"task_id": "easy_triage"}).json()

# 2. Take an action
action = {
    "action_type": "classify_severity",
    "severity": "high",
    "severity_reasoning": "Multiple users affected, login broken"
}
result = requests.post(f"{BASE}/step", json={"task_id": "easy_triage", "action": action}).json()
print(f"Reward: {result['reward']['value']}, Done: {result['done']}")</pre>
  </div>
  <div class="card">
    <h2>📖 Docs</h2>
    <p>Full interactive API docs available at <a href="/docs">/docs</a> (Swagger UI)</p>
  </div>
</div>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
