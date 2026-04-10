"""
inference.py — Bug Triage OpenEnv agent
Connects to the FastAPI server at localhost:7860 and runs a full triage episode.
"""
import requests
import json
import time
import sys

ENV_URL = "http://localhost:7860"
TASK_ID = "easy_triage"


# ─── Server Readiness ─────────────────────────────────────────────────────────

def wait_for_server(retries: int = 20, delay: int = 5) -> None:
    """Poll /health until the FastAPI server is up."""
    print("Waiting for env server to be ready...", flush=True)
    for attempt in range(retries):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=10)
            if r.status_code == 200:
                print(f"[health] Server ready: {r.json()}", flush=True)
                return
        except Exception as e:
            print(f"[health] attempt {attempt+1}/{retries} — {e}", flush=True)
        time.sleep(delay)
    raise RuntimeError("Server never became healthy after retries.")


# ─── Env Calls ────────────────────────────────────────────────────────────────

def reset_env() -> dict:
    """/reset expects JSON body: {"task_id": "easy_triage"}"""
    for attempt in range(5):
        try:
            r = requests.post(
                f"{ENV_URL}/reset",
                json={"task_id": TASK_ID},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[reset] attempt {attempt+1}/5 failed: {e}", flush=True)
            time.sleep(3)
    raise RuntimeError("Failed to reset env after 5 attempts.")


def step_env(action: dict) -> dict:
    """/step expects JSON body: {"task_id": "...", "action": {...}}"""
    for attempt in range(5):
        try:
            r = requests.post(
                f"{ENV_URL}/step",
                json={"task_id": TASK_ID, "action": action},
                timeout=30,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[step] attempt {attempt+1}/5 failed: {e}", flush=True)
            time.sleep(3)
    raise RuntimeError("Failed to step env after 5 attempts.")


# ─── Agent Logic ──────────────────────────────────────────────────────────────

def choose_action(obs: dict, step_count: int) -> dict:
    """
    Simple deterministic agent that triages the bug in 4 steps:
      1. classify_severity
      2. assign_team
      3. add_label
      4. resolve

    Action fields come directly from the Action pydantic model in models.py.
    ActionType values: classify_severity, assign_team, request_info,
                       add_label, resolve, escalate, add_comment
    SeverityLevel: critical, high, medium, low, informational
    TeamName: backend, frontend, infrastructure, security, data, mobile, unknown
    """
    description = obs.get("description", "").lower()
    title = obs.get("title", "").lower()
    text = title + " " + description

    # ── Step 0: Classify severity ──────────────────────────────────────────
    if step_count == 0:
        if any(w in text for w in ["crash", "down", "data loss", "security", "breach"]):
            severity = "critical"
        elif any(w in text for w in ["broken", "major", "fail", "error", "exception"]):
            severity = "high"
        elif any(w in text for w in ["slow", "timeout", "memory", "leak", "degrad"]):
            severity = "medium"
        else:
            severity = "low"

        return {
            "action_type": "classify_severity",
            "severity": severity,
            "severity_reasoning": f"Based on keywords in title/description: '{title}'",
        }

    # ── Step 1: Assign team ────────────────────────────────────────────────
    if step_count == 1:
        if any(w in text for w in ["api", "server", "database", "db", "backend", "memory", "crash"]):
            team = "backend"
        elif any(w in text for w in ["ui", "frontend", "css", "html", "button", "page", "render"]):
            team = "frontend"
        elif any(w in text for w in ["deploy", "infra", "kubernetes", "docker", "network", "port"]):
            team = "infrastructure"
        elif any(w in text for w in ["security", "auth", "login", "token", "xss", "injection"]):
            team = "security"
        elif any(w in text for w in ["data", "pipeline", "etl", "analytics", "ml", "model"]):
            team = "data"
        elif any(w in text for w in ["mobile", "ios", "android", "app"]):
            team = "mobile"
        else:
            team = "backend"

        return {
            "action_type": "assign_team",
            "team": team,
            "team_reasoning": f"Assigned based on keywords in bug description.",
        }

    # ── Step 2: Add labels ─────────────────────────────────────────────────
    if step_count == 2:
        labels = ["bug"]
        if "performance" in text or "slow" in text or "memory" in text:
            labels.append("performance")
        if "crash" in text or "exception" in text:
            labels.append("crash")
        if "security" in text or "auth" in text:
            labels.append("security")
        if "login" in text or "user" in text:
            labels.append("user-facing")

        return {
            "action_type": "add_label",
            "labels": labels,
            "comment": "Auto-labelled during triage.",
        }

    # ── Step 3+: Resolve ───────────────────────────────────────────────────
    return {
        "action_type": "resolve",
        "resolution": "Bug has been triaged: severity classified, team assigned, labels applied.",
        "resolution_type": "fixed",
        "comment": "Triage complete. Assigned to the appropriate team for investigation.",
        "investigation_steps": [
            "Reproduce the issue locally using provided description.",
            "Check recent deployments for regressions.",
            "Review logs and stack traces.",
            "Fix and verify with unit + integration tests.",
        ],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    try:
        # 1. Wait for server
        wait_for_server()

        # 2. Reset environment
        obs = reset_env()
        print(f"\n[reset] Initial observation:", flush=True)
        print(json.dumps(obs, indent=2), flush=True)

        done = False
        total_reward = 0.0
        step_count = 0
        MAX_STEPS = 20  # safety cap

        while not done and step_count < MAX_STEPS:
            # 3. Choose action
            action = choose_action(obs, step_count)
            print(f"\n[step {step_count+1}] Action: {json.dumps(action, indent=2)}", flush=True)

            # 4. Step environment
            result = step_env(action)
            print(f"[step {step_count+1}] Result keys: {list(result.keys())}", flush=True)

            # 5. Parse result
            obs = result.get("observation", {})
            done = result.get("done", False)

            # reward is a dict (Reward.model_dump()) — extract .value
            reward_raw = result.get("reward", {})
            if isinstance(reward_raw, dict):
                reward = float(reward_raw.get("value", 0.0))
                msg = reward_raw.get("message", "")
                print(f"[step {step_count+1}] Reward: {reward} — {msg}", flush=True)
            else:
                reward = float(reward_raw)
                print(f"[step {step_count+1}] Reward: {reward}", flush=True)

            total_reward += reward
            step_count += 1

        print(f"\n{'='*50}", flush=True)
        print(f"Episode complete after {step_count} steps.", flush=True)
        print(f"Total reward: {total_reward:.4f}", flush=True)
        print(f"{'='*50}", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()