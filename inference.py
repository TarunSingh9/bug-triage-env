#!/usr/bin/env python3
"""
Bug Triage OpenEnv — Inference Script
======================================
Rule-based agent that triages bugs without needing an external LLM API.
The agent reads the bug report and applies deterministic rules to:
  1. classify_severity
  2. assign_team
  3. add_label
  4. escalate (when needed)
  5. add_comment

Environment variables (all optional):
  ENV_BASE_URL - Bug Triage environment URL (default: http://localhost:7860)

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
from typing import Any, Dict, List

import requests

# ─── Configuration ────────────────────────────────────────────────────────────

ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
ENV_TIMEOUT  = 60
TASKS        = ["easy_triage", "medium_investigation", "hard_triage"]

# ─── Rule-based Triage Agent ──────────────────────────────────────────────────

SEVERITY_KEYWORDS = {
    "critical": [
        "data loss", "data breach", "security breach", "all users", "down",
        "outage", "breach", "exposed", "leaked", "unauthorized access",
        "production down", "database corrupted", "financial", "gdpr",
    ],
    "high": [
        "login", "authentication", "payment", "charge", "duplicate",
        "500", "crash", "broken", "not working", "regression",
        "150 users", "multiple users", "safari", "cannot", "unable",
    ],
    "medium": [
        "slow", "performance", "timeout", "intermittent", "sometimes",
        "occasional", "delay", "warning", "degraded",
    ],
    "low": [
        "ui", "display", "cosmetic", "typo", "minor", "alignment",
        "color", "font", "style",
    ],
}

TEAM_KEYWORDS = {
    "security": [
        "security", "breach", "unauthorized", "exposed", "leaked",
        "vulnerability", "exploit", "injection", "xss", "csrf",
        "tenant", "data exposure", "gdpr",
    ],
    "backend": [
        "api", "server", "database", "db", "500", "query", "endpoint",
        "payment", "charge", "duplicate charge", "backend", "service",
        "microservice", "queue", "worker",
    ],
    "frontend": [
        "safari", "chrome", "firefox", "browser", "ui", "button",
        "login page", "css", "javascript", "react", "render",
        "display", "visual", "modal", "form",
    ],
    "infrastructure": [
        "deploy", "kubernetes", "docker", "server down", "outage",
        "network", "dns", "ssl", "certificate", "load balancer",
        "memory", "cpu", "disk", "pod", "container",
    ],
    "data": [
        "data loss", "data corruption", "pipeline", "etl", "analytics",
        "report", "dashboard", "metrics", "export", "import",
    ],
    "mobile": [
        "ios", "android", "mobile app", "iphone", "samsung",
        "push notification", "app store", "google play",
    ],
}

ESCALATION_KEYWORDS = [
    "security breach", "data breach", "unauthorized access", "data exposure",
    "tenant", "exposed", "leaked", "financial loss", "gdpr", "compliance",
    "duplicate charge", "payment fraud", "all users affected", "production down",
]

NEEDS_INFO_KEYWORDS = [
    "intermittent", "sometimes", "occasionally", "random", "unclear",
    "unknown", "not sure",
]


def text_contains(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


def get_combined(obs: Dict[str, Any]) -> str:
    return " ".join([
        obs.get("title", ""),
        obs.get("description", ""),
        obs.get("logs", "") or "",
        obs.get("stack_trace", "") or "",
    ])


def classify_severity(obs: Dict[str, Any]) -> str:
    combined = get_combined(obs)
    for severity, keywords in SEVERITY_KEYWORDS.items():
        if text_contains(combined, keywords):
            return severity
    return "medium"


def assign_team(obs: Dict[str, Any]) -> str:
    combined = get_combined(obs)
    for team, keywords in TEAM_KEYWORDS.items():
        if text_contains(combined, keywords):
            return team
    return "backend"


def make_labels(obs: Dict[str, Any]) -> List[str]:
    combined = get_combined(obs).lower()
    labels = []
    label_map = {
        "regression":     ["regression", "worked before", "used to work"],
        "login":          ["login", "sign in", "authentication"],
        "payment":        ["payment", "charge", "billing"],
        "security":       ["security", "breach", "unauthorized", "exposed"],
        "performance":    ["slow", "timeout", "performance", "latency"],
        "safari":         ["safari"],
        "mobile":         ["ios", "android", "mobile"],
        "data-loss":      ["data loss", "data corruption"],
        "production":     ["production", "prod"],
        "needs-info":     ["unclear", "unknown", "intermittent"],
        "duplicate":      ["duplicate charge", "charged twice"],
        "ui":             ["button", "display", "render", "ui", "css"],
        "api":            ["api", "endpoint", "500"],
        "infrastructure": ["deploy", "kubernetes", "outage", "down"],
    }
    for label, keywords in label_map.items():
        if any(kw in combined for kw in keywords):
            labels.append(label)
    return labels[:5] if labels else ["bug", "needs-triage"]


def plan_actions(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Plan the full sequence of actions for this bug report."""
    combined  = get_combined(obs)
    severity  = classify_severity(obs)
    team      = assign_team(obs)
    labels    = make_labels(obs)
    actions   = []

    # Step 1: Request info if report is vague and has no logs
    has_logs  = bool(obs.get("logs") or obs.get("stack_trace"))
    is_vague  = text_contains(combined, NEEDS_INFO_KEYWORDS)
    if is_vague and not has_logs:
        actions.append({
            "action_type": "request_info",
            "needs_info": [
                "steps to reproduce",
                "error logs or stack trace",
                "affected user count",
                "environment details (OS, browser, version)",
            ],
            "comment": "Need more details to properly triage this issue.",
        })

    # Step 2: Classify severity
    actions.append({
        "action_type": "classify_severity",
        "severity": severity,
        "severity_reasoning": (
            f"Classified as {severity} based on keywords indicating "
            f"{'critical' if severity == 'critical' else severity} impact."
        ),
    })

    # Step 3: Assign team
    actions.append({
        "action_type": "assign_team",
        "team": team,
        "team_reasoning": f"Issue relates to {team} domain based on report content.",
    })

    # Step 4: Add labels
    actions.append({
        "action_type": "add_label",
        "labels": labels,
    })

    # Step 5: Escalate if high-impact
    if text_contains(combined, ESCALATION_KEYWORDS):
        escalate_to = (
            "security"
            if text_contains(combined, ["security", "breach", "unauthorized", "exposed", "tenant"])
            else "engineering-lead"
        )
        actions.append({
            "action_type": "escalate",
            "escalate_to": escalate_to,
            "escalation_reason": (
                "High-severity issue with potential security or financial impact "
                "requiring immediate attention."
            ),
            "comment": "Escalating due to critical impact detected in the bug report.",
        })

    # Step 6: Add investigation comment
    actions.append({
        "action_type": "add_comment",
        "comment": (
            f"Triaged as {severity} severity. "
            f"Assigned to {team} team. "
            f"Labels: {', '.join(labels)}."
        ),
        "investigation_steps": [
            "Reproduce the issue in a staging environment",
            "Check recent deployments and changelogs",
            "Review logs and stack traces for root cause",
            "Confirm affected user scope",
            "Apply fix and verify with regression tests",
        ],
    })

    return actions


# ─── Environment Client ───────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._wait_for_env()

    def _wait_for_env(self, max_retries: int = 12, delay: float = 5.0):
        print(f"  Connecting to {self.base_url} ...")
        for i in range(max_retries):
            try:
                r = requests.get(f"{self.base_url}/health", timeout=15)
                if r.status_code == 200:
                    print("  ✓ Environment ready")
                    return
            except Exception:
                pass
            print(f"  Waiting... ({i+1}/{max_retries})")
            time.sleep(delay)
        raise RuntimeError(
            f"Environment at {self.base_url} not reachable after "
            f"{max_retries} retries. Set ENV_BASE_URL correctly."
        )

    def reset(self, task_id: str) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=ENV_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def step(self, task_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/step",
            json={"task_id": task_id, "action": action},
            timeout=ENV_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()


# ─── Episode Runner ───────────────────────────────────────────────────────────

def run_episode(
    env_client: EnvClient,
    task_id: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    try:
        obs = env_client.reset(task_id)
    except Exception as e:
        print(f"  ✗ Reset failed: {e}")
        return {
            "task_id": task_id,
            "error": f"Reset failed: {e}",
            "terminal_score": 0.0,
            "success": False,
        }

    print(f"  Bug: {obs.get('bug_id')} — {obs.get('title', '')[:60]}")

    actions   = plan_actions(obs)
    max_steps = obs.get("max_steps", 10)
    actions   = actions[:max_steps]

    step            = 0
    done            = False
    total_reward    = 0.0
    terminal_score  = None
    grade_breakdown = {}
    actions_taken   = []
    errors          = []

    for action in actions:
        if done:
            break
        step += 1
        try:
            if verbose:
                print(f"  Step {step}: {action.get('action_type')}", end="")
                if action.get("severity"):
                    print(f" → {action['severity']}", end="")
                if action.get("team"):
                    print(f" → {action['team']}", end="")
                if action.get("labels"):
                    print(f" → {action['labels']}", end="")

            result       = env_client.step(task_id, action)
            reward_val   = result["reward"]["value"]
            total_reward += reward_val
            done         = result["done"]
            obs          = result["observation"]
            actions_taken.append(action.get("action_type", "unknown"))

            if verbose:
                msg = result["reward"].get("message", "")
                print(f"\n     Reward: {reward_val:+.3f} | {msg[:80]}")

            if result.get("info", {}).get("terminal_score") is not None:
                terminal_score  = result["info"]["terminal_score"]
                grade_breakdown = result["info"].get("grade_breakdown", {})

        except Exception as e:
            errors.append(f"Step {step}: {type(e).__name__}: {e}")
            if verbose:
                print(f"\n     ERROR: {e}")
            continue

    print(f"\n  ✓ Done — {step} steps, reward: {total_reward:.3f}")
    if terminal_score is not None:
        print(f"  Final score: {terminal_score:.3f}")
        for k, v in grade_breakdown.items():
            if k != "total":
                print(f"    {k}: {v:.3f}")

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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bug Triage OpenEnv Inference")
    parser.add_argument("--task",    choices=TASKS, help="Run a specific task only")
    parser.add_argument("--all",     action="store_true", help="Run all tasks")
    parser.add_argument("--output",  default="baseline_results.json")
    parser.add_argument("--env-url", default=ENV_BASE_URL)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    env_url = args.env_url or ENV_BASE_URL
    print("Bug Triage OpenEnv — Rule-Based Inference")
    print(f"Environment: {env_url}")

    env_client   = EnvClient(base_url=env_url)
    tasks_to_run = [args.task] if args.task else TASKS
    all_results  = []
    start_time   = time.time()

    for task_id in tasks_to_run:
        try:
            result = run_episode(env_client, task_id, verbose=args.verbose)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ✗ Task {task_id} crashed: {e}")
            traceback.print_exc()
            all_results.append({
                "task_id":        task_id,
                "error":          str(e),
                "terminal_score": 0.0,
                "success":        False,
            })
        time.sleep(1)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    scores = []
    for r in all_results:
        score  = r.get("terminal_score") or 0.0
        scores.append(score)
        status = "✓" if r.get("success") else "✗"
        print(f"  {status} {r['task_id']:<30} score={score:.3f}")

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  Average score: {avg:.3f}")
    print(f"  Total time:    {elapsed:.1f}s")

    output = {
        "environment": env_url,
        "tasks":       all_results,
        "summary": {
            "average_score":      round(avg, 4),
            "total_time_seconds": round(elapsed, 1),
            "tasks_passed":       sum(1 for r in all_results if r.get("success")),
            "tasks_total":        len(all_results),
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {args.output}")

    return 0 if avg >= 0.5 else 1


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nFATAL ERROR: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
