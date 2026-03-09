# Plan: Autonomous ML Trainer — Claude Code CLI as Agent

## Context
The original ARCHITECTURE.md described a complex multi-agent system (LangGraph, ChromaDB, MLflow, Optuna, Lightning, etc.) that would *replace* Claude Code with programmatic agents. That's the wrong direction.

The correct framing: **Claude Code CLI IS the research agent** (exactly like karpathy's "spin up your Claude/Codex in this repo"). We build the *environment* it operates within — structured memory, auto-generated context, SQLite tracking, and a live web dashboard. This is a genuine enhancement over karpathy without overengineering.

**karpathy's limitation we fix**: no memory between runs (just results.tsv), no UI, no structured context fed back to the agent.
**What we add**: SQLite experiment DB, auto-generated CONTEXT.md the agent reads each iteration, Streamlit progress dashboard, `.pause` file for human-in-the-loop.

---

## Final Architecture

```
autonomous-ml-trainer/
├── PROGRAM.md          ← Claude Code's instructions (the system prompt equivalent)
├── CONTEXT.md          ← Auto-generated each run; agent reads this for memory
├── train.py            ← Editable by Claude Code. nanoGPT on TinyShakespeare, 5-min budget.
├── prepare.py          ← READ ONLY. Downloads + tokenizes dataset once.
├── log_result.py       ← Claude Code calls this after each run to save to DB
├── context_gen.py      ← Called by log_result.py; reads DB → writes CONTEXT.md
├── db.py               ← sqlite3 stdlib wrapper (~60 lines), no ORM
├── dashboard.py        ← Streamlit live dashboard (separate terminal)
├── requirements.txt
├── data/
│   ├── experiments.db  ← Created on first log_result.py call
│   └── input.bin       ← Tokenized dataset (created by prepare.py)
└── .gitignore
```

**Total: 6 Python files + 2 Markdown files.** No LangGraph, no ChromaDB, no MLflow, no Lightning, no SQLModel, no Optuna. Only stdlib + torch + streamlit.

---

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Training budget | Fixed 5-min wall clock (`BUDGET_SECONDS = 300`) | Karpathy-compatible; Claude Code understands it intuitively; experiments are comparable regardless of model size |
| Editing surface | Agent edits `train.py` directly (karpathy style) | Maximum flexibility; agent sees all hyperparams in one file; can make structural changes, not just hyperparam sweeps |
| Git commits | Claude Code does it naturally; `log_result.py` prints a suggested commit message | No hidden side effects from our scripts; Claude Code already commits when given permission |
| CONTEXT.md update | Auto-regenerate at end of every `log_result.py` call | Always fresh for next iteration; no polling; no manual step |
| Dataset | TinyShakespeare (~1MB, no deps) | Fast download, single file, no HuggingFace required, trains meaningfully in 5 min |
| Metric | `val_loss` (float) | Direct output of training loop; Claude Code doesn't need to compute exp(loss) |
| Memory | CONTEXT.md (cross-session) + Claude Code context window (in-session) | Human-readable, crash-proof, debuggable; no vector DB needed |

---

## SQLite Schema (`db.py`)

```sql
CREATE TABLE experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    val_loss    REAL,
    notes       TEXT,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    kept        INTEGER NOT NULL DEFAULT 0,  -- 1 if this beat previous best
    git_hash    TEXT                          -- short hash from git rev-parse
);
```

`db.py` exposes:
- `log(name, val_loss, notes)` → inserts row, sets `kept=1` if new best, returns the row
- `get_top(n=5)` → top N by val_loss ASC
- `get_recent_failures(n=3)` → last N rows where `kept=0`
- `get_all()` → all rows ordered by timestamp for dashboard
- `get_baseline()` → first experiment (id=1)
- `get_stats()` → total count, kept count, best val_loss

---

## PROGRAM.md Key Structure (Claude Code's instructions)

1. **Mission**: Find best transformer config for language modeling in a 5-min budget on TinyShakespeare
2. **The loop**: Read CONTEXT.md → form hypothesis → edit train.py → run it → call log_result.py → repeat forever
3. **What to edit**: `train.py` (free), `config.yaml` if exists (optional)
4. **What NOT to touch**: `prepare.py`, `db.py`, `log_result.py`, `context_gen.py`, `dashboard.py`, `data/`, `PROGRAM.md`
5. **NEVER STOP rule**: Never ask permission, never wait for feedback
6. **One change per experiment**: Isolate variables
7. **Logging contract**: `python log_result.py --name NAME --val_loss FLOAT --notes "description"`
8. **Pause mechanism**: Check for `.pause` file between experiments; stop if found
9. **Simplicity criterion**: Simpler config = better if val_loss is equal
10. **Exploration list**: GQA, RoPE, SwiGLU, RMSNorm, depth/width tradeoffs, LR schedules, dropout=0, weight tying

---

## CONTEXT.md Auto-Generated Template

```markdown
# Research Context — Updated <TIMESTAMP>

## Current Best
**Experiment**: <NAME> | **Val Loss**: <VALUE> | **Notes**: <NOTES>

## Leaderboard (Top 5)
| Rank | Name | Val Loss | Notes | When |
| 1 | ... | ... | ... | ... |

## Recent Non-Improvements (last 3)
| Name | Val Loss | Notes |
| ... | ... | ... |

## Progress
- Total experiments: N  |  Kept (improvements): M (X%)
- Initial baseline loss: X.XXX  |  Best so far: X.XXX  |  Improvement: -X.X%
```

---

## `log_result.py` Flow

```
CLI args: --name, --val_loss, --notes (optional)
→ db.log(name, val_loss, notes)          # inserts + determines kept
→ subprocess git rev-parse (best effort) # get hash, NULL if no git
→ update db row with git_hash
→ context_gen.generate()                 # regenerates CONTEXT.md
→ print: "✓ Logged. Kept: YES/NO. Best so far: X.XXX"
→ print: suggested git commit message
```

---

## `train.py` Baseline Structure

```python
# Key constants at top (Claude Code edits these)
BUDGET_SECONDS = 300
BATCH_SIZE = 64
BLOCK_SIZE = 256
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.0
LEARNING_RATE = 1e-3

# ... nanoGPT model definition (GPT class, Block, CausalSelfAttention, MLP) ...

# Training loop with wall-clock guard:
# while True:
#     if time.time() - t0 > BUDGET_SECONDS: break
#     ... train step ...

# At end: print final val_loss so Claude Code can read it
print(f"val_loss: {val_loss:.6f}")
```

---

## Streamlit Dashboard (`dashboard.py`)

Two sections:
1. **Progress chart**: Plotly scatter — grey dots for `kept=0`, green dots for `kept=1`, step line for running best (replicates karpathy's classic autoresearch chart)
2. **Leaderboard table**: All experiments sorted by val_loss, with name / val_loss / notes / timestamp / kept badge

Auto-refreshes every 30s (`st.rerun()` with `time.sleep(30)`).

---

## requirements.txt (minimal)

```
torch>=2.1.0
numpy>=1.24.0
streamlit>=1.32.0
plotly>=5.18.0
requests>=2.31.0
```

No MLflow, no LangGraph, no ChromaDB, no Optuna, no Lightning, no SQLModel, no GitPython, no Loguru.

---

## User Quick Start (to implement in README)

```bash
git clone <repo> && cd autonomous-ml-trainer
pip install -r requirements.txt
python prepare.py                          # download TinyShakespeare (~5 sec)

# Terminal 1 — start the agent
claude --dangerously-skip-permissions

# Terminal 2 — live dashboard (optional)
streamlit run dashboard.py                 # → http://localhost:8501

# Pause the agent
touch .pause                               # agent stops after current experiment

# Resume
rm .pause && claude --dangerously-skip-permissions
```

---

## What to do with ARCHITECTURE.md

Replace it entirely. The old ATR design is the wrong architecture for this project. The new ARCHITECTURE.md should describe the actual system: Claude Code CLI as agent + platform files.

---

## Implementation Order

1. `db.py` — schema + 6 query functions (stdlib only)
2. `context_gen.py` — reads DB, writes CONTEXT.md
3. `log_result.py` — CLI entrypoint, calls db + context_gen
4. `prepare.py` — download + tokenize TinyShakespeare
5. `train.py` — nanoGPT baseline with 5-min budget guard, prints `val_loss: X.XXXXXX`
6. `PROGRAM.md` — Claude Code's instruction file
7. `dashboard.py` — Streamlit progress chart + leaderboard
8. `requirements.txt` + `.gitignore`
9. Update `ARCHITECTURE.md` to reflect new design

---

## Verification

1. `python prepare.py` → `data/input.bin` created, no errors
2. `python train.py` → prints `val_loss: X.XXXXXX` after ~5 min (or earlier if on CPU with reduced steps)
3. `python log_result.py --name "baseline" --val_loss 1.5 --notes "initial nanoGPT"` → `data/experiments.db` created, `CONTEXT.md` written
4. `python log_result.py --name "test2" --val_loss 1.4 --notes "lower lr"` → CONTEXT.md shows test2 as new best, `kept=1`
5. `streamlit run dashboard.py` → dashboard loads, shows 2 dots on chart (1 grey, 1 green)
6. `claude --dangerously-skip-permissions` → Claude Code reads PROGRAM.md, reads CONTEXT.md, starts experimenting autonomously
