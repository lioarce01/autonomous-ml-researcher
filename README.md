# Architecture — Autonomous ML Researcher

## Concept

**Claude Code/Codex CLI is the research agent.** The project provides the *environment* the agent operates within: structured memory, auto-generated context, SQLite tracking, and a live web dashboard.

This is a genuine enhancement over [karpathy/autoresearch](https://github.com/karpathy/nanoGPT) without overengineering:

---

## File Structure

```
autonomous-ml-trainer/
├── PROGRAM.md          ← Agent's system prompt (instructions for the agent)
├── CONTEXT.md          ← Auto-generated each run; agent reads this for memory
├── train.py            ← Editable by Agent. nanoGPT on TinyShakespeare.
├── prepare.py          ← READ ONLY. Downloads + tokenizes dataset once.
├── log_result.py       ← Agent calls this after each run to save to DB
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

**Dependencies: stdlib + torch + streamlit + plotly + pandas.**


---

## Agent Loop

```
[Agent reads PROGRAM.md on startup]
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
       ↓ Agent reads this
log_result.py --name NAME --val_loss X --notes "..."
  ├─ db.log()           → INSERT into experiments, set kept=1 if new best
  ├─ git rev-parse      → attach short hash (best effort)
  └─ context_gen.generate()
       └─ reads DB → writes CONTEXT.md
            ↓
    Agent reads CONTEXT.md at start of next iteration
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
| Tokenization | Character-level | No tokenizer dependency; simple; |
| Git integration | Best-effort (subprocess) | Doesn't fail if no git repo; suggested messages only |
| DB library | stdlib `sqlite3` | Zero dependencies |

---

## Dashboard

`dashboard.py` progress chart:
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
