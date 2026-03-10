# Autonomous ML Researcher

An environment for **agentic CLI tools** (Claude Code, Codex, etc.) to autonomously run ML training experiments, analyze results, and iterate toward lower validation loss without human intervention.

---

## What it does

The agent reads its instructions from `PROGRAM.md`, trains a small transformer on TinyShakespeare, and iterates experiments in a loop. Each run:

1. Reads `CONTEXT.md` — its persistent memory of what's been tried and what worked
2. Forms a hypothesis and edits `train.py` accordingly
3. Trains for a fixed 5-minute wall-clock budget
4. Logs the result to SQLite, which regenerates `CONTEXT.md` for the next iteration

The agent can change anything in `train.py` — hyperparameters, architecture, optimizer, attention mechanism — one change per experiment to isolate what works.

---

## Architecture

```
PROGRAM.md       Agent instructions (read-only by the agent)
CONTEXT.md       Auto-generated memory: leaderboard, failures, unexplored techniques
train.py         The only file the agent edits
log_result.py    CLI the agent calls after each run → saves to DB, regenerates CONTEXT.md
dashboard.py     Streamlit live dashboard (optional, separate terminal)
data/            Dataset + SQLite experiment database
```

The feedback loop:

```
train.py → prints val_loss
    ↓
log_result.py → writes to SQLite → regenerates CONTEXT.md
    ↓
Agent reads CONTEXT.md at start of next iteration
```

---

## Quick Start

**Prerequisites**: Python 3.11+, CUDA-capable GPU, CUDA 12.8 drivers.

```bash
git clone <repo> && cd autonomous-ml-trainer

# Create venv and install deps (CUDA build of PyTorch)
python -m venv .venv
.venv/Scripts/activate                           # Windows
pip install -r requirements.txt

# Download and tokenize the dataset (one time)
python prepare.py

# Terminal 1 — start the agent (any agentic CLI)
claude --dangerously-skip-permissions   # Claude Code
# codex                                 # OpenAI Codex CLI

# Terminal 2 — live dashboard (optional)
streamlit run dashboard.py                       # → http://localhost:8501
```

**Pause / resume the agent between experiments:**
```bash
echo "" > .pause    # pause
del .pause          # resume
```
