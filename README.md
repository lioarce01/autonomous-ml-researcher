# Autonomous ML Researcher

An environment for **agentic CLI tools** (Claude Code, Codex, etc.) to autonomously run ML training experiments, analyze results, and iterate toward lower validation loss — no human intervention required.

> **Dataset**: FineWeb-Edu (educational web text, ~400M train tokens at N_SAMPLES=500k). Scale to H100 by increasing N_SAMPLES in `prepare.py`.

---

## What it does

The agent reads its instructions from `PROGRAM.md`, trains a nanoGPT-style transformer on FineWeb-Edu, and iterates in a loop. Each experiment:

1. Reads `CONTEXT.md` — persistent memory: leaderboard, recent failures, unexplored techniques
2. Forms a hypothesis, edits `train.py` (one change only), runs a 5-minute training run
3. Logs the result + hypothesis to SQLite, regenerates `CONTEXT.md` for the next iteration

The agent can modify anything in `train.py` — hyperparameters, architecture, optimizer, LR schedule — always one change per experiment to isolate causality.

---

## Architecture

```
PROGRAM.md       Agent instructions: the loop, exploration guide, strict rules
CONTEXT.md       Auto-generated: leaderboard, failure streak, unexplored techniques
NOTES.md         Agent's persistent research notebook (survives CONTEXT.md regeneration)
MEMORY_LOG.md    Append-only GPU VRAM anomaly log (leak, spike, fragmentation detection)
train.py         The only file the agent edits
prepare.py       Downloads FineWeb-Edu, trains BPE tokenizer, writes data/ (run once)
log_result.py    CLI: --name --val_bpb --notes --hypothesis → SQLite + CONTEXT.md
context_gen.py   Reads DB, writes CONTEXT.md (includes memory anomalies)
db.py            SQLite wrapper (stdlib only, no ORM)
dashboard.py     Streamlit live dashboard (separate terminal)
data/            Dataset + experiments.db (gitignored — regenerate locally)
```

Feedback loop:

```
Agent edits train.py → uv run python train.py → prints val_bpb
    ↓
uv run python log_result.py → SQLite → CONTEXT.md regenerated
    ↓
Agent reads CONTEXT.md → forms next hypothesis
```

---

## Dataset & metric

- **Dataset**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) — educational web text, diverse and high-quality. N_SAMPLES=500k (~400M tokens) for RTX 5070; increase to 5M for H100.
- **Tokenizer**: BPE (HuggingFace `tokenizers`), vocab=8192, ByteLevel pre-tokenizer (GPT-2 style).
- **Metric**: `val_bpb` (bits per byte) = `val_loss_nats / (avg_bytes_per_token × ln2)`. Vocabulary-size independent and comparable across tokenizer changes.
- **Binary format**: 16-byte header (magic + version + vocab_size + n_tokens) + uint16 tokens. Read with `np.memmap` for O(1) memory.

---

## Baseline model

~12M parameter transformer on FineWeb-Edu (BPE vocab=8192).

Already in `train.py`: RoPE, **QK-Norm**, Flash Attention 3 (SDPA fallback on Windows),
**Sliding window SSSL** (64-token local window, every 4th layer global), ReGLU MLP (ReLU²),
MQA (N_KV_HEAD=1), RMSNorm, logit soft-capping (cap=15), **Trapezoidal LR schedule**
(50% warmdown to 0), **per-param-group LR** (embed × 3.0), Muon optimizer, bfloat16 + TF32.

The agent explores on top of this: higher LR (QK-Norm enables the push), nGPT, SWA,
depth/width tradeoffs, WARMDOWN_FRAC tuning, EMBED_LR_MULT tuning, and Tier 5 novel techniques.

---

## Quick Start

**Prerequisites**: Python 3.11+, CUDA-capable GPU, CUDA 12.8 drivers, [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo> && cd autonomous-ml-trainer

# Create venv and install deps (CUDA 12.8 build of PyTorch)
uv venv .venv
uv pip install -r requirements.txt

# Download FineWeb-Edu, train BPE tokenizer, write data/ (one time, ~30-60 min first run)
uv run python prepare.py

# Terminal 1 — start any agentic tool pointed at this directory
claude --dangerously-skip-permissions   # Claude Code
# codex                                 # OpenAI Codex CLI
# cursor                                # Cursor agent mode
# opencode                              # OpenCode
# gemini                                # Gemini CLI
# ... any tool that can read files, run shell commands, and loop

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

- **Agent-agnostic** — works with any tool that can read files and run shell commands: Claude Code, Codex CLI, Cursor, OpenCode, Gemini CLI, or anything equivalent.
- **No magic dependencies** — stdlib + torch + streamlit + tokenizers + datasets only. No LangGraph, MLflow, ChromaDB, Optuna.
- **One change per experiment** — strict rule enforced by PROGRAM.md. Causality over speed.
- **Hypothesis tracking** — every experiment logs a prediction; the agent reflects on what was confirmed or falsified to build a sensitivity model over time.
