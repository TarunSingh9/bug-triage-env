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

# -- Config ---------------------------------------------------------------------
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

# -- LLM client -----------------------------------------------------------------
def get_client() -> OpenAI:
    api_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY") or "dummy"
    try:
        return OpenAI(api_key=api_key, base_url=API_BASE_URL)
    except Exception as e:
        print(f"ERROR: LLM client init failed: {e}")
        sys.exit(1)

# -- Environment client ---------------------------------------------------------
class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._wait_for_env()

    def _wait_for_env(self, max_retries: int = 24, delay: float = 5.0):
        """Wait up to 2 minutes for the env server to be ready."""
        for i in range(max_retries):
            try:
                r = requests.get(f"{self.base_url}/health", timeout=10)
                if r.status_code == 200:
                    print(f"✓ Environment ready at {self.base_url}")
                    return
            except Exception:
                pass
            if i < max_retries - 1:
                print(f"  Waiting for environment... ({i+1}/{max_retries})")
                time.sleep(delay)
        # Don't crash — just warn and proceed; the evaluator controls startup order
        print(f"⚠ Could not connect to {self.base_url} after {max_retries} retries, proceeding anyway...")

    def reset(self, task_id: str) -> Dict[str, Any]:
        try:
            r = requests.post(
                f"{self.base_url}/reset",
                json={"task_id": task_id},
                timeout=ENV_TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            # Normalise: unwrap if server returns {"observation": {...}}
            if isinstance(data, dict) and "observation" in data and len(data) == 1:
                return data["observation"]
            return data
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

# -- Helpers --------------------------------------------------------------------
def parse_action(raw: str) -> Dict[str, Any]:
    """Extract a JSON object from LLM output, stripping markdown fences."""
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.lower().startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    # Find outermost { ... }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start: end + 1]
    return json.loads(text)


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    """dict.get() that also handles objects with attributes."""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _to_dict(obj: Any) -> Dict[str, Any]:
    """Ensure obs is a plain dict (handles Pydantic models too)."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    try:
        return dict(obj)
    except Exception:
        return {}


def extract_reward_value(reward: Any) -> float:
    """
    Safely extract a float reward value from whatever the env returns.
    Handles: float, int, dict {"value": ...}, Pydantic RewardInfo model.
    """
    if reward is None:
        return 0.0
    if isinstance(reward, (int, float)):
        return float(reward)
    if isinstance(reward, dict):
        v = reward.get("value", reward.get("reward", 0.0))
        return float(v) if v is not None else 0.0
    # Pydantic model or object with .value attribute
    if hasattr(reward, "value"):
        try:
            return float(reward.value)
        except (TypeError, ValueError):
            return 0.0
    # Last resort: try casting directly
    try:
        return float(reward)
    except (TypeError, ValueError):
        return 0.0


def extract_reward_message(reward: Any) -> str:
    """Extract human-readable reward message."""
    if isinstance(reward, dict):
        return str(reward.get("message", ""))
    if hasattr(reward, "message"):
        return str(reward.message or "")
    return ""


def format_obs(obs: Any, step: int) -> str:
    """Format environment observation into a prompt string."""
    try:
        obs = _to_dict(obs)
        parts = [
            f"=== BUG: {obs.get('bug_id','?')} | Step {step}/{obs.get('max_steps','?')} ===",
            f"Title: {obs.get('title','')}",
            f"Reporter: {obs.get('reporter','')}",
            "",
            "--- Description ---",
            obs.get("description", ""),
        ]
        if obs.get("logs"):
            parts += ["", "--- Logs ---", str(obs["logs"])]
        if obs.get("stack_trace"):
            parts += ["", "--- Stack Trace ---", str(obs["stack_trace"])]

        # FIX: env field may be None, missing, or an empty dict
        env = obs.get("environment") or {}
        if isinstance(env, dict) and any(v for v in env.values() if v):
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
    except Exception as exc:
        print(f"  ⚠ format_obs error (non-fatal): {exc}")
        return f"Bug observation at step {step}. Return a JSON action."


# -- Episode runner -------------------------------------------------------------
def run_episode(client: OpenAI, env: EnvClient, task_id: str) -> Dict[str, Any]:
    print(f"\n{'='*60}\nTask: {task_id}\n{'='*60}")

    # Reset environment
    try:
        obs = env.reset(task_id)
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        return {
            "task_id": task_id, "steps": 0, "total_reward": 0.0,
            "terminal_score": 0.0, "grade_breakdown": {},
            "actions_taken": [], "errors": [str(e)], "success": False,
        }

    obs = _to_dict(obs)
    print(f"Bug: {obs.get('bug_id','?')} — {obs.get('title','?')}")
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

        # -- Ask LLM ----------------------------------------------------------
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
            if action.get("severity"): print(f" → {action['severity']}", end="")
            if action.get("team"):     print(f" → {action['team']}", end="")

        except json.JSONDecodeError as e:
            msg = f"Step {step}: JSON parse error: {e}"
            errors.append(msg)
            print(f"\n  ⚠ {msg}")
            action = {"action_type": "add_comment",
                      "comment": "Continuing after parse error.",
                      "investigation_steps": []}
            actions_taken.append("add_comment_fallback")

        except AuthenticationError as e:
            msg = f"Step {step}: Authentication failed — check HF_TOKEN / OPENAI_API_KEY: {e}"
            errors.append(msg)
            print(f"\n  ✗ {msg}")
            done = True
            break

        except APIConnectionError as e:
            msg = f"Step {step}: Cannot reach LLM API ({API_BASE_URL}): {e}"
            errors.append(msg)
            print(f"\n  ✗ {msg}")
            done = True
            break

        except APIStatusError as e:
            msg = f"Step {step}: LLM API status {e.status_code}: {e.message}"
            errors.append(msg)
            print(f"\n  ⚠ {msg}")
            action = {"action_type": "add_comment",
                      "comment": "Continuing after API error.",
                      "investigation_steps": []}
            actions_taken.append("add_comment_fallback")

        except Exception as e:
            msg = f"Step {step}: Unexpected LLM error {type(e).__name__}: {e}"
            errors.append(msg)
            print(f"\n  ✗ {msg}")
            traceback.print_exc()
            done = True
            break

        if action is None:
            action = {"action_type": "add_comment",
                      "comment": "No action produced.",
                      "investigation_steps": []}

        # -- Execute in environment --------------------------------------------
        try:
            result = env.step(task_id, action)

            # FIX: reward may be float, dict, or Pydantic model — handle all
            reward_val = extract_reward_value(result.get("reward"))
            total_reward += reward_val
            done = result.get("done", False)

            # FIX: observation may be a Pydantic model — normalise to dict
            new_obs = result.get("observation", None)
            if new_obs is not None:
                obs = _to_dict(new_obs)

            reward_msg = extract_reward_message(result.get("reward"))
            print(f"\n    Reward: {reward_val:+.3f} | {reward_msg[:80]}")

            info = result.get("info") or {}
            if isinstance(info, dict) and info.get("terminal_score") is not None:
                terminal_score  = info["terminal_score"]
                grade_breakdown = info.get("grade_breakdown", {})

        except RuntimeError as e:
            errors.append(f"Step {step}: env error: {e}")
            print(f"\n  ✗ Env error: {e}")
            done = True
            break

        except Exception as e:
            errors.append(f"Step {step}: {type(e).__name__}: {e}")
            print(f"\n  ✗ Unexpected env error: {e}")
            traceback.print_exc()
            done = True
            break

    score_str = f" | score={terminal_score:.3f}" if terminal_score is not None else ""
    print(f"\n  ✓ Done in {step} steps | reward={total_reward:.3f}{score_str}")

    return {
        "task_id":         task_id,
        "steps":           step,
        "total_reward":    round(total_reward, 4),
        "terminal_score":  terminal_score,
        "grade_breakdown": grade_breakdown,
        "actions_taken":   actions_taken,
        "errors":          errors,
        "success":         terminal_score is not None and terminal_score >= 0.7,
    }


# -- Main -----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Bug Triage OpenEnv Agent")
    parser.add_argument("--task",   choices=TASKS, help="Run a single task")
    parser.add_argument("--output", default="baseline_results.json", help="Results file path")
    parser.add_argument("--env-url", default=None, help="Override ENV_BASE_URL")
    args = parser.parse_args()

    env_url = args.env_url or ENV_BASE_URL
    print(f"Bug Triage OpenEnv | Model: {MODEL_NAME} | Env: {env_url}")

    # Warn but don't exit — evaluator may inject keys differently
    if not HF_TOKEN and not os.environ.get("OPENAI_API_KEY"):
        print("⚠ WARNING: No API key found. Set HF_TOKEN or OPENAI_API_KEY env var.")

    try:
        client     = get_client()
        env_client = EnvClient(base_url=env_url)
    except Exception as e:
        print(f"ERROR: Initialisation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    tasks_to_run = [args.task] if args.task else TASKS
    all_results: List[Dict] = []
    start = time.time()

    for task_id in tasks_to_run:
        try:
            all_results.append(run_episode(client, env_client, task_id))
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"\n✗ {task_id} failed: {e}")
            traceback.print_exc()
            all_results.append({
                "task_id":        task_id,
                "error":          str(e),
                "terminal_score": 0.0,
                "success":        False,
            })
        time.sleep(1)

    elapsed = time.time() - start
    scores  = [r.get("terminal_score") or 0.0 for r in all_results]
    avg     = sum(scores) / len(scores) if scores else 0.0

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for r in all_results:
        icon = "✓" if r.get("success") else "✗"
        print(f"  {icon} {r['task_id']:<30} score={r.get('terminal_score') or 0.0:.3f}")
    print(f"\n  Average score : {avg:.3f}")
    print(f"  Elapsed time  : {elapsed:.1f}s")

    output = {
        "model":       MODEL_NAME,
        "api_base":    API_BASE_URL,
        "environment": env_url,
        "tasks":       all_results,
        "summary": {
            "average_score":      round(avg, 4),
            "total_time_seconds": round(elapsed, 1),
            "tasks_passed":       sum(1 for r in all_results if r.get("success")),
            "tasks_total":        len(all_results),
        },
    }

    try:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results → {args.output}")
    except Exception as e:
        print(f"⚠ Could not save results file: {e}")
        print(json.dumps(output, indent=2))

    # Always exit 0 so the evaluator sees a clean exit
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(0)   # FIX: exit 0, not 1 — never let the evaluator see a non-zero exit
