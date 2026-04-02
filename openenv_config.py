"""Task definitions config."""

TASK_DEFS = {
    "easy_triage": {
        "name": "Basic Bug Severity Classification",
        "difficulty": "easy",
        "description": "Given a clear bug report with obvious symptoms, correctly classify severity and assign to the right team within 3 steps.",
        "max_steps": 5,
        "success_threshold": 0.7,
    },
    "medium_investigation": {
        "name": "Incomplete Bug Report Investigation",
        "difficulty": "medium",
        "description": "Handle an incomplete bug report: identify missing information, request it, then classify and route correctly after receiving it.",
        "max_steps": 8,
        "success_threshold": 0.75,
    },
    "hard_triage": {
        "name": "Ambiguous Multi-System Incident",
        "difficulty": "hard",
        "description": "Triage a complex incident involving multiple systems, partial logs, conflicting severity signals, and a security implication.",
        "max_steps": 10,
        "success_threshold": 0.85,
    },
}
