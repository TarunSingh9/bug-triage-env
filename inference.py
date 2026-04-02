#!/usr/bin/env python3
"""
Bug Triage OpenEnv — Baseline Inference Script
================================================
Runs an LLM agent against all 3 tasks using the OpenAI client.

Environment variables required:
  API_BASE_URL   - API base URL for the LLM
  MODEL_NAME     - Model identifier
  HF_TOKEN       - HuggingFace / API key (used as auth bearer)
  ENV_BASE_URL   - (optional) Bug Triage environment URL, defaults to localhost:7860

Usage:
  python inference.py
  python inference.py --task easy_triage
  python inference.py --all --output results.json
"""
import os
import sys
import json
import time
import argparse
import traceback
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI


# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# Timeout for environment requests
ENV_TIMEOUT = 30

TASKS = ["easy_triage", "medium_investigation", "hard_triage"]


# ─── Agent System Prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert software engineering triage agent. Your job is to analyze bug reports and take appropriate actions to classify, route, and address them.

You operate in an environment that accepts specific JSON actions. At each step, you must return EXACTLY one JSON action object.

Available action types and their required fields:

1. classify_severity — Assign a severity level
   {"action_type": "classify_severity", "severity": "<critical|high|medium|low|informational>", "severity_reasoning": "<why>"}

2. assign_team — Route to the correct engineering team
   {"action_type": "assign_team", "team": "<backend|frontend|infrastructure|security|data|mobile|unknown>", "team_reasoning": "<why>"}

3. request_info — Ask for missing information (use when bug report lacks key details)
   {"action_type": "request_info", "needs_info": ["<info1>", "<info2>"], "comment": "<what you need and why>"}

4. add_label — Apply relevant labels/tags
   {"action_type": "add_label", "labels": ["<label1>", "<label2>"]}

5. escalate — Escalate to a specific team or person
   {"action_type": "escalate", "escalate_to": "<who>", "escalation_reason": "<why>", "comment": "<details>"}

6. add_comment — Add investigation notes or next steps
   {"action_type": "add_comment", "comment": "<your analysis>", "investigation_steps": ["<step1>", "<step2>"]}

7. resolve — Mark the bug as resolved (only when you have enough information)
   {"action_type": "resolve", "resolution": "<detailed resolution>", "resolution_type": "<fixed|wont_fix|duplicate|cannot_reproduce|by_design>"}

CRITICAL RULES:
- Return ONLY valid JSON, no markdown, no explanation outside the JSON
- Analyze logs, stack traces, and environment data carefully
- Look for security implications (data exposure, auth issues)
- Consider financial impact (duplicate charges, data loss)
- Be efficient — accomplish the task in as few steps as possible
- For incomplete reports, use request_info FIRST before classifying
"""


# ─── OpenAI Client ────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    return OpenAI(
        api_key=HF_TOKEN or "dummy",
        base_url=API_BASE_URL,
    )


# ─── Environment Client ───────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str = ENV_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._wait_for_env()

    def _wait_for_env(self, max_retries: int = 5, delay: float = 3.0):
        """Wait for the environment to be ready."""
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
        print(f"⚠ Could not connect to environment at {self.base_url}, proceeding anyway...")

    def reset(self, task_id: str) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=ENV_TIMEOUT
        )
        r.raise_for_status()
        return r.json()

    def step(self, task_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/step",
            json={"task_id": task_id, "action": action},
            timeout=ENV_TIMEOUT
        )
        r.raise_for_status()
        return r.json()

    def get_tasks(self) -> List[Dict[str, Any]]:
        r = requests.get(f"{self.base_url}/tasks", timeout=ENV_TIMEOUT)
        r.raise_for_status()
        return r.json()["tasks"]


# ─── Agent Logic ──────────────────────────────────────────────────────────────

def format_observation(obs: Dict[str, Any], step: int) -> str:
    """Format observation as a prompt for the LLM."""
    parts = [
        f"=== BUG REPORT: {obs.get('bug_id')} ===",
        f"Title: {obs.get('title')}",
        f"Reporter: {obs.get('reporter')}",
        f"Created: {obs.get('created_at')}",
        f"Step: {step}/{obs.get('max_steps')}",
        "",
        "--- Description ---",
        obs.get('description', ''),
    ]

    if obs.get('logs'):
        parts.extend(["", "--- Logs ---", obs.get('logs', '')])

    if obs.get('stack_trace'):
        parts.extend(["", "--- Stack Trace ---", obs.get('stack_trace', '')])

    env = obs.get('environment', {})
    if any(env.values()):
        parts.append("\n--- Environment ---")
        for k, v in env.items():
            if v is not None:
                parts.append(f"  {k}: {v}")

    # Current state
    parts.append("\n--- Current Triage State ---")
    parts.append(f"  Severity: {obs.get('current_severity') or 'Not set'}")
    parts.append(f"  Team: {obs.get('current_team') or 'Not assigned'}")
    parts.append(f"  Labels: {obs.get('current_labels') or []}")
    parts.append(f"  Escalated: {obs.get('is_escalated', False)}")

    if obs.get('history'):
        parts.append(f"\n--- History ({len(obs['history'])} steps) ---")
        for h in obs['history'][-3:]:  # Last 3 steps
            parts.append(f"  Step {h['step']}: {h['action_type']} (reward: {h['reward']:.3f})")

    parts.append("\n\nReturn your next action as a JSON object.")
    return "\n".join(parts)


def run_agent_step(
    client: OpenAI,
    conversation: List[Dict[str, str]],
    obs: Dict[str, Any],
    step: int
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Run one agent step, return the action and updated conversation."""
    user_message = format_observation(obs, step)
    conversation.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation,
        temperature=0.1,
        max_tokens=500,
    )

    raw = response.choices[0].message.content.strip()
    conversation.append({"role": "assistant", "content": raw})

    # Parse JSON action
    # Strip markdown code blocks if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    action = json.loads(raw)
    return action, conversation


def run_episode(
    client: OpenAI,
    env_client: EnvClient,
    task_id: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run a full episode for one task. Returns results dict."""
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    # Reset environment
    obs = env_client.reset(task_id)
    print(f"Bug: {obs['bug_id']} — {obs['title']}")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    step = 0
    done = False
    total_reward = 0.0
    terminal_score = None
    grade_breakdown = {}
    actions_taken = []
    errors = []

    while not done:
        step += 1
        try:
            action, conversation = run_agent_step(client, conversation, obs, step)
            actions_taken.append(action.get("action_type", "unknown"))

            if verbose:
                print(f"\n  Step {step}: {action.get('action_type', '?')}", end="")
                if action.get('severity'):
                    print(f" → severity={action['severity']}", end="")
                if action.get('team'):
                    print(f" → team={action['team']}", end="")
                if action.get('labels'):
                    print(f" → labels={action['labels']}", end="")

            result = env_client.step(task_id, action)
            reward_val = result['reward']['value']
            total_reward += reward_val
            done = result['done']
            obs = result['observation']

            if verbose:
                msg = result['reward'].get('message', '')
                print(f"\n     Reward: {reward_val:+.3f} | {msg[:80]}")

            if result.get('info', {}).get('terminal_score') is not None:
                terminal_score = result['info']['terminal_score']
                grade_breakdown = result['info'].get('grade_breakdown', {})

        except json.JSONDecodeError as e:
            errors.append(f"Step {step}: JSON parse error — {e}")
            # Fallback action
            action = {"action_type": "add_comment", "comment": "Agent error: could not parse response"}
            try:
                result = env_client.step(task_id, action)
                done = result['done']
                obs = result['observation']
            except Exception:
                done = True

        except Exception as e:
            errors.append(f"Step {step}: {type(e).__name__}: {e}")
            if verbose:
                print(f"\n     ERROR: {e}")
            done = True

    print(f"\n  ✓ Episode complete in {step} steps")
    print(f"  Total reward: {total_reward:.3f}")
    if terminal_score is not None:
        print(f"  Final score:  {terminal_score:.3f}")
        if grade_breakdown:
            for k, v in grade_breakdown.items():
                if k != "total":
                    print(f"    {k}: {v:.3f}")

    return {
        "task_id": task_id,
        "steps": step,
        "total_reward": round(total_reward, 4),
        "terminal_score": terminal_score,
        "grade_breakdown": grade_breakdown,
        "actions_taken": actions_taken,
        "errors": errors,
        "success": terminal_score is not None and terminal_score >= 0.7,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bug Triage OpenEnv Baseline Inference")
    parser.add_argument("--task", choices=TASKS, help="Run a specific task only")
    parser.add_argument("--all", action="store_true", help="Run all tasks (default)")
    parser.add_argument("--output", default="baseline_results.json", help="Output JSON file")
    parser.add_argument("--env-url", default=ENV_BASE_URL, help="Environment base URL")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    print("Bug Triage OpenEnv — Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print(f"Environment: {args.env_url or ENV_BASE_URL}")

    if not HF_TOKEN and not os.environ.get("OPENAI_API_KEY"):
        print("⚠ WARNING: No API key found. Set HF_TOKEN or OPENAI_API_KEY.")

    # Init clients
    client = get_client()
    env_url = args.env_url or ENV_BASE_URL
    env_client = EnvClient(base_url=env_url)

    tasks_to_run = [args.task] if args.task else TASKS

    all_results = []
    start_time = time.time()

    for task_id in tasks_to_run:
        try:
            result = run_episode(client, env_client, task_id, verbose=args.verbose)
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Task {task_id} failed: {e}")
            traceback.print_exc()
            all_results.append({
                "task_id": task_id,
                "error": str(e),
                "terminal_score": 0.0,
                "success": False,
            })
        time.sleep(1)  # Rate limit courtesy

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    scores = []
    for r in all_results:
        score = r.get('terminal_score') or 0.0
        scores.append(score)
        status = "✓" if r.get('success') else "✗"
        print(f"  {status} {r['task_id']:<30} score={score:.3f}")

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  Average score: {avg:.3f}")
    print(f"  Total time:    {elapsed:.1f}s")

    # Save results
    output = {
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "environment": env_url,
        "tasks": all_results,
        "summary": {
            "average_score": round(avg, 4),
            "total_time_seconds": round(elapsed, 1),
            "tasks_passed": sum(1 for r in all_results if r.get('success')),
            "tasks_total": len(all_results),
        }
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {args.output}")

    return 0 if avg >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
