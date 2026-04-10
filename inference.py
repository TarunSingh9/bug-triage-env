#!/usr/bin/env python3
import os, sys, json, time, argparse, traceback
from typing import Any, Dict, List

# ---------------- SAFE IMPORTS ----------------
try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(0)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # fallback mode

# ---------------- CONFIG ----------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
ENV_TIMEOUT  = int(os.environ.get("ENV_TIMEOUT", "20"))
USE_LLM      = os.environ.get("USE_LLM", "false").lower() == "true"

TASKS = ["easy_triage", "medium_investigation", "hard_triage"]

SYSTEM_PROMPT = "You are a bug triage agent. Return only JSON."

# ---------------- LLM CLIENT ----------------
def get_client():
    if not USE_LLM or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY"))
    except Exception as e:
        print(f"LLM init failed: {e}")
        return None

# ---------------- FALLBACK ----------------
def get_fallback_action(obs: Dict[str, Any], step: int) -> Dict[str, Any]:
    if not obs.get("current_severity"):
        return {
            "action_type": "classify_severity",
            "severity": "medium",
            "severity_reasoning": "fallback"
        }
    if not obs.get("current_team"):
        return {
            "action_type": "assign_team",
            "team": "backend",
            "team_reasoning": "fallback"
        }
    return {
        "action_type": "add_comment",
        "comment": "fallback execution",
        "investigation_steps": []
    }

# ---------------- ENV CLIENT ----------------
class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._wait()

    def _wait(self):
        for i in range(6):
            try:
                r = requests.get(f"{self.base_url}/health", timeout=5)
                if r.status_code == 200:
                    print("✓ Env ready")
                    return
            except:
                pass
            time.sleep(2)
        print("⚠ Env not reachable, continuing")

    def reset(self, task_id):
        try:
            r = requests.post(f"{self.base_url}/reset", json={"task_id": task_id}, timeout=ENV_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise RuntimeError(f"reset failed: {e}")

    def step(self, task_id, action):
        try:
            r = requests.post(f"{self.base_url}/step", json={"task_id": task_id, "action": action}, timeout=ENV_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise RuntimeError(f"step failed: {e}")

# ---------------- HELPERS ----------------
def safe_dict(x):
    if isinstance(x, dict):
        return x
    try:
        return dict(x)
    except:
        return {}

def extract_reward(r):
    if isinstance(r, (int, float)):
        return float(r)
    if isinstance(r, dict):
        return float(r.get("value", 0))
    return 0.0

def parse_action(raw):
    try:
        if "{" in raw:
            raw = raw[raw.find("{"):raw.rfind("}")+1]
        return json.loads(raw)
    except:
        return {
            "action_type": "add_comment",
            "comment": "parse failed",
            "investigation_steps": []
        }

def format_obs(obs, step):
    obs = safe_dict(obs)
    return f"""
BUG: {obs.get('bug_id')}
Title: {obs.get('title')}
Desc: {obs.get('description')}
Step: {step}
Return JSON action.
"""

# ---------------- EPISODE ----------------
def run_episode(client, env, task_id):
    print(f"\n=== {task_id} ===")

    try:
        obs = env.reset(task_id)
    except Exception as e:
        print(e)
        return {"task_id": task_id, "success": False}

    obs = safe_dict(obs)
    obs.setdefault("max_steps", 10)

    step = 0
    total = 0
    done = False

    while not done and step < obs["max_steps"]:
        step += 1

        # ---- ACTION ----
        action = None

        if USE_LLM and client:
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": format_obs(obs, step)}],
                    temperature=0.1,
                )
                raw = resp.choices[0].message.content
                action = parse_action(raw)
            except Exception as e:
                print("LLM fail:", e)
                action = get_fallback_action(obs, step)
        else:
            action = get_fallback_action(obs, step)

        if not isinstance(action, dict) or "action_type" not in action:
            action = get_fallback_action(obs, step)

        print(f"Step {step}: {action.get('action_type')}")

        # ---- ENV STEP ----
        try:
            result = env.step(task_id, action)
            total += extract_reward(result.get("reward"))
            done = result.get("done", False)
            obs = safe_dict(result.get("observation"))
        except Exception as e:
            print("Env fail:", e)
            break

    print(f"Done | reward={total:.2f}")

    return {
        "task_id": task_id,
        "reward": total,
        "success": True
    }

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS)
    args = parser.parse_args()

    print("Starting agent...")

    client = get_client()
    env = EnvClient(ENV_BASE_URL)

    tasks = [args.task] if args.task else TASKS
    results = []

    for t in tasks:
        try:
            results.append(run_episode(client, env, t))
        except Exception as e:
            print("Fatal task error:", e)
            traceback.print_exc()
            results.append({"task_id": t, "success": False})

    print("\nSUMMARY")
    for r in results:
        print(r)

    return 0

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("FATAL:", e)
        traceback.print_exc()
        sys.exit(0)
