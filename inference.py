import requests
import json
import time
import sys

ENV_URL = "http://localhost:7860"  # or wherever your HF Space container listens

def reset_env():
    for attempt in range(5):
        try:
            r = requests.post(f"{ENV_URL}/reset", timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[reset] attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(3)
    raise RuntimeError("Failed to reset env")

def step_env(action):
    for attempt in range(5):
        try:
            r = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[step] attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(3)
    raise RuntimeError("Failed to step env")

def main():
    try:
        obs = reset_env()
        print("Initial obs:", obs, flush=True)

        done = False
        total_reward = 0.0

        while not done:
            # 👇 Replace this with your actual agent logic
            action = 0  

            result = step_env(action)
            obs = result.get("observation", result.get("obs", {}))
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            total_reward += reward

        print(f"Episode complete. Total reward: {total_reward}", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"FATAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
