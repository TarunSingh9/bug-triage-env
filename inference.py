import requests
import json
import time
import sys

ENV_URL = "http://localhost:7860"
TASK_ID = "easy_triage"


def wait_for_server(retries=15, delay=5):
    """Wait until the FastAPI server is up before doing anything."""
    print("Waiting for env server to be ready...", flush=True)
    for attempt in range(retries):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=10)
            if r.status_code == 200:
                print(f"Server ready: {r.json()}", flush=True)
                return True
        except Exception as e:
            print(f"[health] attempt {attempt+1}/{retries} failed: {e}", flush=True)
        time.sleep(delay)
    raise RuntimeError("Server never became ready after retries")


def reset_env():
    for attempt in range(5):
        try:
            # /reset expects JSON body with task_id
            r = requests.post(
                f"{ENV_URL}/reset",
                json={"task_id": TASK_ID},
                timeout=30
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[reset] attempt {attempt+1}/5 failed: {e}", flush=True)
            time.sleep(3)
    raise RuntimeError("Failed to reset env after 5 attempts")


def step_env(action: dict):
    """
    action must be a dict matching your Action model, e.g.:
    {"action_type": "label", "label": "bug", "comment": "..."}
    Adjust keys to match your actual Action pydantic model.
    """
    for attempt in range(5):
        try:
            r = requests.post(
                f"{ENV_URL}/step",
                json={"task_id": TASK_ID, "action": action},
                timeout=30
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[step] attempt {attempt+1}/5 failed: {e}", flush=True)
            time.sleep(3)
    raise RuntimeError("Failed to step env after 5 attempts")


def get_tasks():
    """Fetch available tasks to understand what's expected."""
    try:
        r = requests.get(f"{ENV_URL}/tasks", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[tasks] Could not fetch tasks: {e}", flush=True)
        return {}


def choose_action(obs: dict) -> dict:
    """
    Your agent logic here.
    obs is the observation dict returned by reset/step.
    Must return a dict matching your Action pydantic model.

    Look at your models.py Action class for the exact required fields.
    Example structure — adjust to match your actual Action model:
    """
    # Safely extract whatever fields your observation has
    bug_report = obs.get("bug_report", obs.get("description", ""))
    
    # Simple heuristic agent — replace with your real logic / LLM call
    action = {
        "action_type": "label",   # adjust to your Action model's field names
        "label": "bug",
        "comment": "Triaged based on description.",
        "priority": "medium",
        "assignee": None,
    }
    return action


def main():
    try:
        # Step 1: Wait for the server to be healthy before calling it
        wait_for_server()

        # Step 2 (optional): Log available tasks
        tasks = get_tasks()
        print(f"Available tasks: {json.dumps(tasks, indent=2)}", flush=True)

        # Step 3: Reset the environment
        obs = reset_env()
        print(f"Initial observation: {json.dumps(obs, indent=2)}", flush=True)

        done = False
        total_reward = 0.0
        step_count = 0
        MAX_STEPS = 100  # safety limit to avoid infinite loop

        while not done and step_count < MAX_STEPS:
            # Step 4: Choose an action based on observation
            action = choose_action(obs)
            print(f"[step {step_count+1}] Action: {action}", flush=True)

            # Step 5: Step the environment
            result = step_env(action)
            print(f"[step {step_count+1}] Result: {json.dumps(result, indent=2)}", flush=True)

            obs = result.get("observation", {})
            
            # reward may be a dict (model_dump of your Reward model) or a float
            reward_raw = result.get("reward", 0.0)
            if isinstance(reward_raw, dict):
                reward = reward_raw.get("value", reward_raw.get("score", 0.0))
            else:
                reward = float(reward_raw)

            done = result.get("done", False)
            total_reward += reward
            step_count += 1

        print(f"\nEpisode complete after {step_count} steps.", flush=True)
        print(f"Total reward: {total_reward}", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
