---
title: Bug Triage OpenEnv
emoji: 🐛
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Real-world bug triage environment for AI agent training
---

# 🐛 Bug Triage OpenEnv

A real-world **software engineering environment** where AI agents learn to triage bug reports — the daily work of every software engineering team.

## What It Is

Bug triage is the process of:
1. Reading a bug report
2. Classifying its **severity** (critical → informational)
3. Assigning to the correct **team** (backend, frontend, infrastructure, security...)
4. Adding appropriate **labels** 
5. Requesting **missing information** when needed
6. **Escalating** high-impact issues (security breaches, data loss, financial harm)
7. Providing an **investigation plan**

This environment simulates exactly that, with 3 tasks of increasing difficulty.

---

## 🎯 Tasks

| Task | Difficulty | Description | Max Steps |
|------|-----------|-------------|-----------|
| `easy_triage` | 🟢 Easy | Clear Safari regression bug. Classify severity, assign team, add labels. | 5 |
| `medium_investigation` | 🟡 Medium | Incomplete payment failure report. Request missing info, discover a duplicate charge, escalate. | 8 |
| `hard_triage` | 🔴 Hard | Multi-system incident with tenant data exposure. Identify the security vector, escalate to security, provide investigation plan. | 10 |

---

## 📡 API Reference

### Reset Environment
```http
POST /reset
Content-Type: application/json

{"task_id": "easy_triage"}
```

**Returns:** Initial `Observation` object

### Execute Action
```http
POST /step
Content-Type: application/json

{
  "task_id": "easy_triage",
  "action": {
    "action_type": "classify_severity",
    "severity": "high",
    "severity_reasoning": "Login broken for 150+ Safari users after deploy"
  }
}
```

**Returns:** `{observation, reward, done, info}`

### Get State
```http
GET /state?task_id=easy_triage
```

### List Tasks
```http
GET /tasks
```

---

## 🎮 Action Space

| Action | Required Fields | When to Use |
|--------|----------------|-------------|
| `classify_severity` | `severity` | Always — classify as critical/high/medium/low/informational |
| `assign_team` | `team` | Always — route to backend/frontend/infrastructure/security/data/mobile |
| `add_label` | `labels` | Always — add descriptive tags |
| `request_info` | `needs_info` | When key info is missing from the report |
| `escalate` | `escalate_to`, `escalation_reason` | When severity warrants executive/security/on-call attention |
| `add_comment` | `comment`, `investigation_steps` | To document analysis and next steps |
| `resolve` | `resolution`, `resolution_type` | When bug can be closed |

---

## 🏆 Reward Function

The reward function provides **dense signal** throughout the episode:

| Signal | Description |
|--------|-------------|
| +0.30 | Correct severity classification |
| +0.25 | Correct team assignment |
| +0.25 | Correct escalation (when needed) |
| +0.20 | Info gathering (medium task) |
| +0.15 | Label quality (F1 score) |
| +0.10 | Investigation quality |
| -0.15 | Unnecessary info requests |
| -0.20 | Unnecessary escalation |
| -0.20 | Resolving before classifying |
| +bonus | Terminal: ±0.25 based on final score |

---

## 🚀 Quick Start

```python
import requests

BASE = "https://your-space.hf.space"

# Reset
obs = requests.post(f"{BASE}/reset", json={"task_id": "easy_triage"}).json()
print(obs["title"])  # "Login button not working on Safari 16..."

# Step 1: Classify severity
result = requests.post(f"{BASE}/step", json={
    "task_id": "easy_triage",
    "action": {"action_type": "classify_severity", "severity": "high"}
}).json()
print(result["reward"]["value"])  # ~0.3

# Step 2: Assign team
result = requests.post(f"{BASE}/step", json={
    "task_id": "easy_triage",
    "action": {"action_type": "assign_team", "team": "frontend"}
}).json()

# Step 3: Add labels
result = requests.post(f"{BASE}/step", json={
    "task_id": "easy_triage",
    "action": {"action_type": "add_label", "labels": ["safari", "regression", "login", "v2.4.1"]}
}).json()
print(result["done"])  # True
print(result["info"]["terminal_score"])  # ~0.95
```

---

## 🧪 Running the Baseline

```bash
# Set environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_BASE_URL="https://your-space.hf.space"

# Run baseline
python inference.py

# Output: baseline_results.json
```

**Expected baseline scores** (gpt-4o-mini):
- easy_triage: ~0.85
- medium_investigation: ~0.65
- hard_triage: ~0.55

---

## 🏗️ Local Setup

```bash
git clone https://github.com/your-username/bug-triage-env
cd bug-triage-env

pip install -r requirements.txt
python app.py  # Starts on port 7860
```

Or with Docker:
```bash
docker build -t bug-triage-env .
docker run -p 7860:7860 bug-triage-env
```

---

## 📁 Project Structure

```
bug-triage-env/
├── app.py                 # FastAPI application (main entrypoint)
├── models.py              # Pydantic typed models (Observation, Action, Reward)
├── openenv.yaml           # OpenEnv spec metadata
├── openenv_config.py      # Task definitions
├── inference.py           # Baseline inference script
├── requirements.txt
├── Dockerfile
├── envs/
│   └── bug_triage_env.py  # Core environment logic
├── tasks/
│   └── scenarios.py       # Bug report scenarios + ground truth
├── graders/
│   └── task_graders.py    # Deterministic task graders
└── tests/
    └── test_env.py        # Unit tests
```

---

## License

MIT
