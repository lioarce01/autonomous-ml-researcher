# Autonomous ML Researcher

An environment for **agentic CLI tools** (Claude Code, Codex, etc.) to autonomously run ML training experiments, analyze results, and iterate toward lower validation loss — no human intervention required.

---

## What it does

The agent reads its instructions from `PROGRAM.md`, trains a nanoGPT-style transformer on TinyShakespeare, and iterates in a loop. Each experiment:

1. Reads `CONTEXT.md` — persistent memory: leaderboard, recent failures, unexplored techniques
2. Forms a hypothesis, edits `train.py` (one change only), runs a 5-minute training run
3. Logs the result + hypothesis to SQLite, regenerates `CONTEXT.md` for the next iteration

The agent can modify anything in `train.py` — hyperparameters, architecture, optimizer, LR schedule — always one change per experiment to isolate causality.

---

## Architecture

```
PROGRAM.md       Agent instructions: the loop, exploration guide, strict rules
CONTEXT.md       Auto-generated: leaderboard, failure streak, unexplored techniques
train.py         The only file the agent edits
prepare.py       Downloads TinyShakespeare, tokenizes to data/input.bin (run once)
log_result.py    CLI: --name --val_loss --notes --hypothesis → SQLite + CONTEXT.md
context_gen.py   Reads DB, writes CONTEXT.md
db.py            SQLite wrapper (stdlib only, no ORM)
dashboard.py     Streamlit live dashboard (separate terminal)
data/            Dataset + experiments.db
```

Feedback loop:

```
Agent edits train.py → uv run python train.py → prints val_loss
    ↓
uv run python log_result.py → SQLite → CONTEXT.md regenerated
    ↓
Agent reads CONTEXT.md → forms next hypothesis
```

---

## Baseline model

~22M parameter character-level transformer on TinyShakespeare (65-char vocab, ~1M chars).

Already in `train.py`: RoPE, Flash Attention, SwiGLU MLP, MQA (N_KV_HEAD=1), RMSNorm,
logit soft-capping, WSD LR schedule, bfloat16 + TF32.

The agent explores on top of this: Muon optimizer, QK-Norm, nGPT, SWA, trapezoidal LR,
depth/width tradeoffs, gradient clipping removal, and more.

---

## Quick Start

**Prerequisites**: Python 3.11+, CUDA-capable GPU, CUDA 12.8 drivers, [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo> && cd autonomous-ml-trainer

# Create venv and install deps (CUDA 12.8 build of PyTorch)
uv venv .venv
uv pip install -r requirements.txt

# Download and tokenize the dataset (one time)
uv run python prepare.py

# Terminal 1 — start the agent
claude --dangerously-skip-permissions

# Terminal 2 — live dashboard (optional)
streamlit run dashboard.py    # → http://localhost:8501
```

**Pause / resume between experiments:**
```bash
echo "" > .pause    # pause after current experiment finishes
rm .pause           # resume
```

---

## Design principles

- **No magic dependencies** — stdlib + torch + streamlit only. No LangGraph, MLflow, ChromaDB, Optuna.
- **One change per experiment** — strict rule enforced by PROGRAM.md. Causality over speed.
- **Hypothesis tracking** — every experiment logs a prediction; the agent reflects on what was confirmed or falsified to build a sensitivity model over time.
