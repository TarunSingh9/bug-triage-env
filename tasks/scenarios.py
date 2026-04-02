"""
Realistic bug report scenarios for the Bug Triage environment.
Each task has a detailed scenario with ground truth answers.
"""
from typing import Any, Dict

# ─── Task 1: Easy — Clear severity, obvious team ──────────────────────────────
EASY_TASK: Dict[str, Any] = {
    "task_id": "easy_triage",
    "bug": {
        "bug_id": "BUG-4821",
        "title": "Login button not working on Safari 16 — users can't sign in",
        "description": (
            "Our support team has received 12 tickets in the past hour from users on Safari 16.x "
            "reporting that clicking the 'Sign In' button on the login page has no effect. "
            "The button appears clickable but nothing happens. Users on Chrome and Firefox are unaffected. "
            "This started approximately 1.5 hours ago, around the same time as our v2.4.1 frontend deploy.\n\n"
            "Reproduction steps:\n"
            "1. Open login.example.com on Safari 16\n"
            "2. Enter valid credentials\n"
            "3. Click 'Sign In'\n"
            "4. Expected: redirect to dashboard\n"
            "5. Actual: nothing happens, no error shown"
        ),
        "reporter": "sarah.ops@example.com",
        "created_at": "2024-01-15T14:32:00Z",
        "logs": (
            "[14:30:01] Safari/605.1.15 GET /login 200\n"
            "[14:30:02] TypeError: undefined is not a function (evaluating 'e.preventDefault')\n"
            "    at handleLoginSubmit (login.bundle.js:1:48291)\n"
            "[14:30:02] Unhandled Promise Rejection: Cannot read property 'submit' of null\n"
            "[14:30:45] 12 identical errors in last 60s\n"
        ),
        "stack_trace": (
            "TypeError: undefined is not a function\n"
            "  at handleLoginSubmit (login.bundle.js:1:48291)\n"
            "  at HTMLButtonElement.onclick (login.html:1)\n"
        ),
        "environment": {
            "os": "macOS 13.x",
            "browser": "Safari 16.2",
            "app_version": "v2.4.1",
            "region": "us-east-1",
            "service": "web-frontend",
            "user_count_affected": 150,
            "error_rate": 0.08,
        }
    },
    "ground_truth": {
        "severity": "high",
        "team": "frontend",
        "required_labels": ["safari", "regression", "login", "v2.4.1"],
        "expected_action_sequence": ["classify_severity", "assign_team", "add_label"],
        "resolution_hints": ["regression introduced in v2.4.1 deploy", "Safari-specific JS compatibility issue"],
        "should_escalate": False,
        "needs_info": False,
        "security_issue": False,
    }
}

# ─── Task 2: Medium — Missing info, needs investigation ──────────────────────
MEDIUM_TASK: Dict[str, Any] = {
    "task_id": "medium_investigation",
    "bug": {
        "bug_id": "BUG-5503",
        "title": "Users reporting 'payment failed' errors intermittently",
        "description": (
            "We've had 3 users reach out via Twitter and 2 via support email saying their payments "
            "are failing. The errors seem to happen 'sometimes'. One user said it worked on the second try. "
            "Another user said they were charged but the order didn't go through.\n\n"
            "No internal alerts have fired. The payments dashboard shows normal success rates."
        ),
        "reporter": "customer_success@example.com",
        "created_at": "2024-01-15T16:10:00Z",
        "logs": "",
        "stack_trace": "",
        "environment": {
            "os": None,
            "browser": None,
            "app_version": None,
            "region": None,
            "service": None,
            "user_count_affected": 5,
            "error_rate": None,
        }
    },
    "info_responses": {
        "payment_provider_logs": (
            "Stripe webhook logs (last 2h):\n"
            "  - 3 payment_intent.payment_failed events\n"
            "  - Failure reason: 'card_declined' (2x), 'insufficient_funds' (1x)\n"
            "  - 1 duplicate charge detected: charge_abc123 created twice within 2s\n"
            "  - Webhook delivery retries: 7 (normal: <2)\n"
        ),
        "affected_user_ids": (
            "Affected users: u_8821, u_3302, u_7710\n"
            "All three used 3D Secure cards\n"
            "All failures occurred within a 20-minute window: 15:48–16:08 UTC\n"
            "One user (u_3302) was charged $89.99 twice — potential duplicate charge"
        ),
        "recent_deploys": (
            "Recent deployments:\n"
            "  - 2024-01-15 15:30 UTC: payments-service v3.2.1 → v3.2.2 (changelog: 'idempotency key refactor')\n"
            "  - 2024-01-14 09:00 UTC: frontend v2.4.0 → v2.4.1\n"
        ),
    },
    "ground_truth": {
        "severity": "critical",  # Duplicate charge = potential financial harm
        "team": "backend",
        "required_labels": ["payments", "regression", "duplicate-charge", "3ds", "v3.2.2"],
        "expected_action_sequence": ["request_info", "classify_severity", "assign_team", "escalate"],
        "should_escalate": True,
        "escalate_to": "payments-oncall",
        "needs_info": True,
        "required_info_fields": ["payment_provider_logs", "affected_user_ids", "recent_deploys"],
        "security_issue": False,
        "financial_impact": True,
    }
}

# ─── Task 3: Hard — Multi-system, security implications ───────────────────────
HARD_TASK: Dict[str, Any] = {
    "task_id": "hard_triage",
    "bug": {
        "bug_id": "BUG-6001",
        "title": "Unusual API response times + some users seeing other users' data",
        "description": (
            "Our monitoring shows API p99 latency spiked from 120ms to 3,400ms starting 40 minutes ago. "
            "Simultaneously, two enterprise customers have separately reported (via their CSMs) that they "
            "briefly saw what appeared to be data belonging to a different organization in their dashboard — "
            "account names, project titles, and one user claims to have seen an invoice.\n\n"
            "One customer specifically mentioned their org ID in the URL looked correct, but the data shown "
            "was not theirs. This resolved itself on page refresh.\n\n"
            "DB CPU is at 94% (normal: 15–20%). Redis cache hit rate dropped from 94% to 31%.\n"
            "No recent deploys in the past 6 hours."
        ),
        "reporter": "sre-alert-bot@example.com",
        "created_at": "2024-01-15T18:05:00Z",
        "logs": (
            "[17:24:01] WARN: Cache miss storm detected — 8,200 cache misses/min (baseline: 900/min)\n"
            "[17:24:15] ERROR: DB connection pool exhausted (pool_size=50, waiting=127)\n"
            "[17:25:02] ERROR: query timeout after 5000ms — SELECT * FROM projects WHERE org_id=?\n"
            "[17:26:11] WARN: Redis key expiry bulk event — 14,000 keys expired simultaneously\n"
            "[17:26:45] ERROR: [org_id=acme-corp] returned data from cache key 'projects:tenantX'\n"
            "[17:26:45] ERROR: Cache key collision detected — tenant isolation may be compromised\n"
            "[17:27:00] CRITICAL: 2 tenant cross-contamination events logged\n"
        ),
        "stack_trace": (
            "CacheKeyCollisionError: Key 'projects:2024-01' resolved to wrong tenant\n"
            "  at TenantCache.get (cache.service.ts:142)\n"
            "  at ProjectsController.list (projects.controller.ts:88)\n"
            "  at Layer.handle [as handle_request] (router/layer.js:95)\n"
            "\n"
            "Caused by: Redis SCAN returned stale tenant_id binding after mass expiry event\n"
        ),
        "environment": {
            "os": "Linux (production cluster)",
            "browser": None,
            "app_version": "api-service v5.1.3",
            "region": "us-east-1, eu-west-1",
            "service": "api-service, cache-layer, postgres-primary",
            "user_count_affected": 2800,
            "error_rate": 0.23,
        }
    },
    "ground_truth": {
        "severity": "critical",
        "team": "infrastructure",  # Cache layer + DB, but also security
        "secondary_team": "security",  # Must also involve security due to data exposure
        "required_labels": [
            "critical", "security", "data-leak", "tenant-isolation",
            "cache", "performance", "incident"
        ],
        "expected_action_sequence": [
            "classify_severity",
            "escalate",       # Must escalate to security AND SRE
            "assign_team",
            "add_label",
            "add_comment",    # Investigation steps
        ],
        "should_escalate": True,
        "escalate_to": "security-oncall",  # Security escalation is required
        "needs_info": False,
        "security_issue": True,           # Data exposure between tenants
        "required_security_identification": True,
        "investigation_steps_keywords": [
            "tenant isolation", "cache", "redis", "rollback", "audit log",
            "affected tenants", "data exposure", "invalidate cache"
        ],
        "financial_impact": True,
    }
}

ALL_TASKS = {
    "easy_triage": EASY_TASK,
    "medium_investigation": MEDIUM_TASK,
    "hard_triage": HARD_TASK,
}
