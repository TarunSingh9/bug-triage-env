#!/usr/bin/env python3
import os, sys, json, time, argparse, traceback
from typing import Any, Dict, List, Tuple

try:
    import requests
except ImportError:
    print("ERROR: pip install requests"); sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai"); sys.exit(1)

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
ENV_TIMEOUT  = 30
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
- Return ONLY valid JSON
- For incomplete reports, use request_info FIRST
- Look for security and financial implications
"""

def get_client() -> OpenAI:
    try:
        return OpenAI(api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY","dummy"), base_url=API_BASE_URL)
    except Exception as e:
        print(f"ERROR: LLM client failed: {e}"); sys.exit(1)

class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._wait_for_env()

    def _wait_for_env(self, max_retries=10, delay=5.0):
        for i in range(max_retries):
            try:
                r = requests.get(f"{self.base_url}/health", timeout=10)
                if r.status_code == 200:
                    print(f"? Environment ready at {self.base_url}"); return
            except Exception:
                pass
            if i < max_retries - 1:
                print(f"  Waiting for environment... ({i+1}/{max_retries})")
                time.sleep(delay)
        print(f"? Could not connect to {self.base_url}, proceeding anyway...")

    def reset(self, task_id: str) -> Dict[str, Any]:
        try:
            r = requests.post(f"{self.base_url}/reset", json={"task_id": task_id}, timeout=ENV_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Reset timed out after {ENV_TIMEOUT}s")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Cannot connect to environment: {e}")
        except requests.exceptions.HTTPError:
            raise RuntimeError(f"Reset failed HTTP {r.status_code}: {r.text}")

    def step(self, task_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        try:
            r = requests.post(f"{self.base_url}/step", json={"task_id": task_id, "action": action}, timeout=ENV_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Step timed out after {ENV_TIMEOUT}s")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Cannot connect to environment: {e}")
        except requests.exceptions.HTTPError:
            raise RuntimeError(f"Step failed HTTP {r.status_code}: {r.text}")

def parse_action(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"): part = part[4:].strip()
            if part.startswith("{"): text = part; break
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end+1]
    return json.loads(text)

def format_obs(obs: Dict[str, Any], step: int) -> str:
    try:
        parts = [
            f"=== BUG: {obs.get('bug_id','?')} | Step {step}/{obs.get('max_steps','?')} ===",
            f"Title: {obs.get('title','')}", f"Reporter: {obs.get('reporter','')}",
            "", "--- Description ---", obs.get('description',''),
        ]
        if obs.get('logs'): parts += ["", "--- Logs ---", obs['logs']]
        if obs.get('stack_trace'): parts += ["", "--- Stack Trace ---", obs['stack_trace']]
        env = obs.get('environment') or {}
        if any(env.values()):
            parts.append("--- Environment ---")
            parts += [f"  {k}: {v}" for k,v in env.items() if v]
        parts += [
            "--- Triage State ---",
            f"  Severity: {obs.get('current_severity') or 'Not set'}",
            f"  Team: {obs.get('current_team') or 'Not assigned'}",
            f"  Labels: {obs.get('current_labels') or []}",
            f"  Escalated: {obs.get('is_escalated', False)}",
        ]
        history = obs.get('history') or []
        if history:
            parts.append(f"--- History ({len(history)} steps) ---")
            parts += [f"  Step {h.get('step')}: {h.get('action_type')} (reward: {h.get('reward',0):.3f})" for h in history[-3:]]
        parts.append("\nReturn your next action as a JSON object.")
        return "\n".join(parts)
    except Exception:
        return f"Bug observation at step {step}. Return a JSON action."

def run_episode(client: OpenAI, env: EnvClient, task_id: str) -> Dict[str, Any]:
    print(f"\n{'='*60}\nTask: {task_id}\n{'='*60}")
    try:
        obs = env.reset(task_id)
    except Exception as e:
        print(f"? Reset failed: {e}")
        return {"task_id":task_id,"steps":0,"total_reward":0.0,"terminal_score":0.0,"actions_taken":[],"errors":[str(e)],"success":False}

    print(f"Bug: {obs.get('bug_id','?')} — {obs.get('title','?')}")
    conversation = [{"role":"system","content":SYSTEM_PROMPT}]
    step, done, total_reward = 0, False, 0.0
    terminal_score, grade_breakdown = None, {}
    actions_taken, errors = [], []
    max_steps = obs.get("max_steps", 15)

    while not done and step < max_steps + 5:
        step += 1
        action = None

        # Get action from LLM
        try:
            conversation.append({"role":"user","content":format_obs(obs, step)})
            resp = client.chat.completions.create(model=MODEL_NAME, messages=conversation, temperature=0.1, max_tokens=500)
            raw = resp.choices[0].message.content.strip()
            conversation.append({"role":"assistant","content":raw})
            action = parse_action(raw)
            actions_taken.append(action.get("action_type","unknown"))
            print(f"\n  Step {step}: {action.get('action_type','?')}", end="")
            if action.get("severity"): print(f" ? {action['severity']}", end="")
            if action.get("team"):     print(f" ? {action['team']}", end="")
        except json.JSONDecodeError as e:
            errors.append(f"Step {step}: JSON parse error: {e}")
            action = {"action_type":"add_comment","comment":"Parse error, continuing.","investigation_steps":[]}
            actions_taken.append("add_comment_fallback")
        except RuntimeError as e:
            errors.append(f"Step {step}: LLM error: {e}")
            action = {"action_type":"add_comment","comment":"LLM error, continuing.","investigation_steps":[]}
            actions_taken.append("add_comment_fallback")
        except Exception as e:
            errors.append(f"Step {step}: {type(e).__name__}: {e}")
            done = True; break

        # Execute action in environment
        try:
            result = env.step(task_id, action)
            reward_val = result.get("reward",{}).get("value", 0.0)
            total_reward += reward_val
            done = result.get("done", False)
            obs  = result.get("observation", obs)
            print(f"\n    Reward: {reward_val:+.3f} | {str(result.get('reward',{}).get('message',''))[:80]}")
            info = result.get("info") or {}
            if info.get("terminal_score") is not None:
                terminal_score  = info["terminal_score"]
                grade_breakdown = info.get("grade_breakdown", {})
        except RuntimeError as e:
            errors.append(f"Step {step}: env error: {e}")
            done = True; break
        except Exception as e:
            errors.append(f"Step {step}: {type(e).__name__}: {e}")
            done = True; break

    print(f"\n  ? Done in {step} steps | reward={total_reward:.3f}", end="")
    if terminal_score is not None: print(f" | score={terminal_score:.3f}", end="")
    print()
    return {
        "task_id":task_id, "steps":step, "total_reward":round(total_reward,4),
        "terminal_score":terminal_score, "grade_breakdown":grade_breakdown,
        "actions_taken":actions_taken, "errors":errors,
        "success": terminal_score is not None and terminal_score >= 0.7,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--output", default="baseline_results.json")
    parser.add_argument("--env-url", default=None)
    args = parser.parse_args()

    env_url = args.env_url or ENV_BASE_URL
    print(f"Bug Triage OpenEnv | Model: {MODEL_NAME} | Env: {env_url}")
    if not HF_TOKEN and not os.environ.get("OPENAI_API_KEY"):
        print("? WARNING: No API key set. Set HF_TOKEN or OPENAI_API_KEY.")

    try:
        client     = get_client()
        env_client = EnvClient(base_url=env_url)
    except Exception as e:
        print(f"ERROR: Init failed: {e}"); sys.exit(1)

    tasks_to_run = [args.task] if args.task else TASKS
    all_results, start = [], time.time()

    for task_id in tasks_to_run:
        try:
            all_results.append(run_episode(client, env_client, task_id))
        except KeyboardInterrupt:
            print("\nInterrupted."); break
        except Exception as e:
            print(f"\n? {task_id} failed: {e}"); traceback.print_exc()
            all_results.append({"task_id":task_id,"error":str(e),"terminal_score":0.0,"success":False})
        time.sleep(1)

    elapsed = time.time() - start
    scores  = [r.get("terminal_score") or 0.0 for r in all_results]
    avg     = sum(scores)/len(scores) if scores else 0.0

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for r in all_results:
        print(f"  {'?' if r.get('success') else '?'} {r['task_id']:<30} score={r.get('terminal_score') or 0.0:.3f}")
    print(f"\n  Average: {avg:.3f} | Time: {elapsed:.1f}s")

    output = {
        "model":MODEL_NAME, "api_base":API_BASE_URL, "environment":env_url,
        "tasks":all_results,
        "summary":{"average_score":round(avg,4),"total_time_seconds":round(elapsed,1),
                   "tasks_passed":sum(1 for r in all_results if r.get("success")),"tasks_total":len(all_results)}
    }
    try:
        with open(args.output,"w") as f: json.dump(output,f,indent=2)
        print(f"  Results ? {args.output}")
    except Exception as e:
        print(f"? Could not save results: {e}")
        print(json.dumps(output, indent=2))

    return 0 if avg >= 0.5 else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted."); sys.exit(130)
    except Exception as e:
        print(f"\nFATAL: {type(e).__name__}: {e}"); traceback.print_exc(); sys.exit(1)