from __future__ import annotations
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ValidationError
from typing import Any, Dict

from models import Action
from envs.bug_triage_env import BugTriageEnvironment

app = FastAPI(title="Bug Triage OpenEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_envs: Dict[str, BugTriageEnvironment] = {}

def _get_env(task_id: str) -> BugTriageEnvironment:
    if task_id not in _envs:
        _envs[task_id] = BugTriageEnvironment(task_id=task_id)
    return _envs[task_id]

class ResetRequest(BaseModel):
    task_id: str = "easy_triage"

class StepRequest(BaseModel):
    task_id: str = "easy_triage"
    action: Dict[str, Any]

@app.get("/health")
async def health():
    return {"status": "ok", "service": "bug-triage-openenv", "version": "1.0.0"}

@app.get("/tasks")
async def list_tasks():
    try:
        from tasks.scenarios import ALL_TASKS
        from openenv_config import TASK_DEFS
        return {"tasks": [{"id": tid, **TASK_DEFS.get(tid, {})} for tid in ALL_TASKS]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset(req: ResetRequest):
    try:
        return _get_env(req.task_id).reset().model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")

@app.post("/step")
async def step(req: StepRequest):
    try:
        try:
            action = Action(**req.action)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=f"Invalid action: {e}")
        obs, reward, done, info = _get_env(req.task_id).step(action)
        return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())

@app.get("/state")
async def state(task_id: str = "easy_triage"):
    try:
        return _get_env(task_id).state().model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def home():
    return "<h1>Bug Triage OpenEnv</h1><p><a href='/docs'>API Docs</a></p>"

def main():
    """Required entry point for multi-mode deployment."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()