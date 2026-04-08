#!/usr/bin/env python3
import os, sys, json, time, argparse, traceback
from typing import Any, Dict, List

try:
    import requests
except ImportError:
    print("ERROR: pip install requests"); sys.exit(1)

try:
    from openai import OpenAI, AuthenticationError, APIConnectionError, APIStatusError
except ImportError:
    print("ERROR: pip install openai"); sys.exit(1)

# \u2500\u2500 Config \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
ENV_TIMEOUT  = int(os.environ.get("ENV_TIMEOUT", "30"))
TASKS        = ["easy_triage", "medium_investigation", "hard_triage"]

SYSTEM_PROMPT = """You are an expert software engineering triage agent. Analyze bug reports and take actions.
Return EXACTLY one JSON action object per step. No markdown, no explanation outside JSON.

Actions:
1. {"action_type":"classify_severity","severity":"<critical|high|medium|low|informational>","severity_reasoning":"<why>"}
2. {"action_type":"assign_team","team":"<backend|frontend|infrastructure|security|data|mobile|unknown>","team_reasoning":"<why>"}
3. {"action_type":"request_info","needs_info":["<info1>"],"comment":"<why>"}
4. {"action_type":"add_label","labels":["<label1>"]}
5. {"action_type":"escalate","escalate_to":"<who>","escalation_reason":"<why>","comment":"<details>"}
6. {"action_type":"add_comment","comment":"<analysis>","investigation_steps":["<step1>"]}
7. {"action_type":"resolve","resolution":"<resolution>","resolution_type":"<fixed|wont_fix|duplicate|cannot_reproduce|by_design>"}

Rules:
- Return ONLY valid JSON, no markdown fences, no extra text
- For incomplete reports, use request_info FIRST
- Look for security and financial implications
- Always classify severity before assigning team
"""

# \u2500\u2500 LLM client \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def get_client() -> OpenAI:
    api_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY") or "dummy"
    try:
        return OpenAI(api_key=api_key, base_url=API_BASE_URL)
    except Exception as e:
        print(f"ERROR: LLM client init failed: {e}")
        sys.exit(1)

# \u2500\u2500 Environment client \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._wait_for_env()

    def _wait_for_env(self, max_retries: int = 12, delay: float = 5.0):
        for i in range(max_retries):
            try:
                r = requests.get(f"{self.base_url}/health", timeout=10)
                if r.status_code == 200:
                    print(f"\u2713 Environment ready at {self.base_url}")
                    return
            except Exception:
                pass
            if i < max_retries - 1:
                print(f"  Waiting for environment... ({i+1}/{max_retries})")
                time.sleep(delay)
        print(f"\u26a0 Could not connect to {self.base_url} after {max_retries} retries, proceeding anyway...")

    def reset(self, task_id: str) -> Dict[str, Any]:
        try:
            r = requests.post(
                f"{self.base_url}/reset",
                json={"task_id": task_id},
                timeout=ENV_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Reset timed out after {ENV_TIMEOUT}s")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Cannot connect to environment: {e}")
        except requests.exceptions.HTTPError:
            raise RuntimeError(f"Reset failed HTTP {r.status_code}: {r.text[:200]}")

    def step(self, task_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        try:
            r = requests.post(
                f"{self.base_url}/step",
                json={"task_id": task_id, "action": action},
                timeout=ENV_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Step timed out after {ENV_TIMEOUT}s")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Cannot connect to environment: {e}")
        except requests.exceptions.HTTPError:
            raise RuntimeError(f"Step failed HTTP {r.status_code}: {r.text[:200]}")

# \u2500\u2500 Helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def parse_action(raw: str) -> Dict[str, Any]:
    """Extract a JSON object from LLM output, stripping markdown fences."""
    text = raw.strip()
    # Strip ```json ... ``` fences
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    # Find outermost { ... }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def format_obs(obs: Dict[str, Any], step: int) -> str:
    """Format environment observation into a prompt string."""
    try:
        parts = [
            f"=== BUG: {obs.get('bug_id','?')} | Step {step}/{obs.get('max_steps','?')} ===",
            f"Title: {obs.get('title','')}",
            f"Reporter: {obs.get('reporter','')}",
            "",
            "--- Description ---",
            obs.get("description", ""),
        ]
        if obs.get("logs"):
            parts += ["", "--- Logs ---", obs["logs"]]
        if obs.get("stack_trace"):
            parts += ["", "--- Stack Trace ---", obs["stack_trace"]]
        env = obs.get("environment") or {}
        if any(env.values()):
            parts.append("--- Environment ---")
            parts += [f"  {k}: {v}" for k, v in env.items() if v]
        parts += [
            "--- Triage State ---",
            f"  Severity:  {obs.get('current_severity') or 'Not set'}",
            f"  Team:      {obs.get('current_team') or 'Not assigned'}",
            f"  Labels:    {obs.get('current_labels') or []}",
            f"  Escalated: {obs.get('is_escalated', False)}",
        ]
        history: List[Dict] = obs.get("history") or []
        if history:
            parts.append(f"--- History ({len(history)} steps) ---")
            parts += [
                f"  Step {h.get('step')}: {h.get('action_type')} (reward: {h.get('reward', 0):.3f})"
                for h in history[-3:]
            ]
        parts.append("\nReturn your next action as a single JSON object only.")
        return "\n".join(parts)
    except Exception:
        return f"Bug observation at step {step}. Return a JSON action."


# \u2500\u2500 Episode runner \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def run_episode(client: OpenAI, env: EnvClient, task_id: str) -> Dict[str, Any]:
    print(f"\n{'='*60}\nTask: {task_id}\n{'='*60}")

    # Reset environment
    try:
        obs = env.reset(task_id)
    except Exception as e:
        print(f"\u2717 Reset failed: {e}")
        return {
            "task_id": task_id, "steps": 0, "total_reward": 0.0,
            "terminal_score": 0.0, "grade_breakdown": {},
            "actions_taken": [], "errors": [str(e)], "success": False,
        }

    print(f"Bug: {obs.get('bug_id','?')} \u2014 {obs.get('title','?')}")
    conversation   = [{"role": "system", "content": SYSTEM_PROMPT}]
    step           = 0
    done           = False
    total_reward   = 0.0
    terminal_score = None
    grade_breakdown: Dict = {}
    actions_taken: List[str] = []
    errors: List[str] = []
    max_steps = obs.get("max_steps", 15)

    while not done and step < max_steps + 5:
        step += 1
        action = None

        # \u2500\u2500 Ask LLM \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        try:
            conversation.append({"role": "user", "content": format_obs(obs, step)})
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                temperature=0.1,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            conversation.append({"role": "assistant", "content": raw})
            action = parse_action(raw)
            actions_taken.append(action.get("action_type", "unknown"))
            print(f"\n  Step {step}: {action.get('action_type','?')}", end="")
            if action.get("severity"): print(f" \u2192 {action['severity']}", end="")
            if action.get("team"):     print(f" \u2192 {action['team']}", end="")

        except json.JSONDecodeError as e:
            msg = f"Step {step}: JSON parse error: {e}"
            errors.append(msg)
            print(f"\n  \u26a0 {msg}")
            action = {"action_type": "add_comment", "comment": "Continuing after parse error.", "investigation_steps": []}
            actions_taken.append("add_comment_fallback")

        except AuthenticationError as e:
            msg = f"Step {step}: Authentication failed \u2014 check HF_TOKEN / OPENAI_API_KEY: {e}"
            errors.append(msg)
            print(f"\n  \u2717 {msg}")
            done = True
            break

        except APIConnectionError as e:
            msg = f"Step {step}: Cannot reach LLM API ({API_BASE_URL}): {e}"
            errors.append(msg)
            print(f"\n  \u2717 {msg}")
            done = True
            break

        except APIStatusError as e:
            msg = f"Step {step}: LLM API status {e.status_code}: {e.message}"
            errors.append(msg)
            print(f"\n  \u26a0 {msg}")
            action = {"action_type": "add_comment", "comment": "Continuing after API error.", "investigation_steps": []}
            actions_taken.append("add_comment_fallback")

        except Exception as e:
            msg = f"Step {step}: Unexpected LLM error {type(e).__name__}: {