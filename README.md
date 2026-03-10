# Architecture — Autonomous ML Trainer

## Concept

**Claude Code CLI is the research agent.** The project provides the *environment* the agent operates within: structured memory, auto-generated context, SQLite tracking, and a live web dashboard.

This is a genuine enhancement over [karpathy/autoresearch](https://github.com/karpathy/nanoGPT) without overengineering:

| Aspect | karpathy/autoresearch | This project |
|---|---|---|
| Agent | Claude/Codex via API | Claude Code CLI (same model, better tool use) |
| Memory | `results.tsv` (no cross-session memory) | SQLite DB + auto-generated `CONTEXT.md` |
| Context | None | `CONTEXT.md` injected each iteration |
| UI | None | Streamlit dashboard with Plotly chart |
| Human control | Kill process | Touch `.pause` file |
| Training budget | Fixed wall-clock | Fixed wall-clock (300s) |
| Editing surface | Agent edits `train.py` | Agent edits `train.py` |

---

## File Structure

```
autonomous-ml-trainer/
├── PROGRAM.md          ← Claude Code's system prompt (instructions for the agent)
├── CONTEXT.md          ← Auto-generated each run; agent reads this for memory
├── train.py            ← Editable by Claude Code. nanoGPT on TinyShakespeare.
├── prepare.py          ← READ ONLY. Downloads + tokenizes dataset once.
├── log_result.py       ← Claude Code calls this after each run to save to DB
├── context_gen.py      ← Called by log_result.py; reads DB → writes CONTEXT.md
├── db.py               ← sqlite3 stdlib wrapper, no ORM
├── dashboard.py        ← Streamlit live dashboard
├── requirements.txt
├── .gitignore
└── data/
    ├── experiments.db  ← Created on first log_result.py call
    ├── input.bin       ← Tokenized dataset (created by prepare.py)
    └── input.txt       ← Raw TinyShakespeare text
```

**6 Python files + 2 Markdown files. No LangGraph, no ChromaDB, no MLflow, no Lightning.**
Dependencies: stdlib + torch + streamlit + plotly + pandas.

---

## Agent Loop

```
[Claude Code reads PROGRAM.md on startup]
    ↓
Loop forever:
    1. Read CONTEXT.md          → what's been tried, what's working
    2. Form hypothesis           → what one change might help?
    3. Edit train.py             → make exactly one meaningful change
    4. python train.py           → 5-min wall-clock training
    5. Read output               → find "val_loss: X.XXXXXX"
    6. python log_result.py ...  → saves to DB, regenerates CONTEXT.md
    7. Check for .pause file     → stop if found, else repeat
```

---

## SQLite Schema

```sql
CREATE TABLE experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    val_loss    REAL,
    notes       TEXT,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    kept        INTEGER NOT NULL DEFAULT 0,  -- 1 if this beat previous best
    git_hash    TEXT
);
```

`kept=1` means this experiment set a new best val_loss at the time it was logged.

---

## Data Flow

```
train.py
  └─ prints "val_loss: X.XXXXXX"
       ↓ Claude Code reads this
log_result.py --name NAME --val_loss X --notes "..."
  ├─ db.log()           → INSERT into experiments, set kept=1 if new best
  ├─ git rev-parse      → attach short hash (best effort)
  └─ context_gen.generate()
       └─ reads DB → writes CONTEXT.md
            ↓
     Claude Code reads CONTEXT.md at start of next iteration
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Training budget | 5-min wall clock | Experiments are comparable regardless of model size |
| Editing surface | Agent edits `train.py` directly | Maximum flexibility; agent can make structural changes |
| Memory | CONTEXT.md (cross-session) | Human-readable, crash-proof, no vector DB |
| Dataset | TinyShakespeare (~1MB) | Fast download, no HuggingFace, meaningful in 5 min |
| Metric | `val_loss` (float) | Direct output; simpler than computing perplexity |
| Tokenization | Character-level | No tokenizer dependency; simple; karpathy-compatible |
| Git integration | Best-effort (subprocess) | Doesn't fail if no git repo; suggested messages only |
| DB library | stdlib `sqlite3` | Zero dependencies |

---

## Dashboard

`dashboard.py` replicates karpathy's autoresearch progress chart:
- **Grey dots**: experiments that didn't improve val_loss
- **Green dots**: experiments that set a new best
- **Green step line**: running best over time
- **Leaderboard table**: all experiments sorted by val_loss

Auto-refreshes every 30 seconds.

---

## Quick Start

```bash
git clone <repo> && cd autonomous-ml-trainer
uv venv                                    # create .venv
uv pip install -r requirements.txt        # install deps
uv run python prepare.py                  # download TinyShakespeare (~5 sec)

# Terminal 1 — start the agent
claude --dangerously-skip-permissions

# Terminal 2 — live dashboard (optional)
uv run streamlit run dashboard.py         # → http://localhost:8501

# Pause the agent between experiments
echo "" > .pause

# Resume
rm .pause && claude --dangerously-skip-permissions
```
