"""
context_gen.py — Reads DB and writes CONTEXT.md for the agent.
"""

import os
from datetime import datetime
import db

CONTEXT_PATH = os.path.join(os.path.dirname(__file__), "CONTEXT.md")


def generate():
    stats = db.get_stats()
    top = db.get_top(5)
    failures = db.get_recent_failures(3)
    baseline = db.get_baseline()

    total = stats["total"] or 0
    kept_count = int(stats["kept_count"] or 0)
    best_val_loss = stats["best_val_loss"]
    pct = f"{kept_count/total*100:.0f}%" if total > 0 else "N/A"

    # Current best
    if top:
        best = top[0]
        best_section = (
            f"**Experiment**: `{best['name']}` | "
            f"**Val Loss**: `{best['val_loss']:.6f}` | "
            f"**Notes**: {best['notes'] or '—'}"
        )
    else:
        best_section = "_No experiments logged yet._"

    # Leaderboard table
    if top:
        header = "| Rank | Name | Val Loss | Notes | When |\n|---|---|---|---|---|\n"
        rows = ""
        for i, exp in enumerate(top, 1):
            rows += (
                f"| {i} | `{exp['name']}` | `{exp['val_loss']:.6f}` "
                f"| {exp['notes'] or '—'} | {exp['timestamp'][:16]} |\n"
            )
        leaderboard = header + rows
    else:
        leaderboard = "_No experiments yet._"

    # Recent non-improvements
    if failures:
        f_header = "| Name | Val Loss | Notes |\n|---|---|---|\n"
        f_rows = ""
        for exp in failures:
            f_rows += (
                f"| `{exp['name']}` | `{exp['val_loss']:.6f}` "
                f"| {exp['notes'] or '—'} |\n"
            )
        failures_section = f_header + f_rows
    else:
        failures_section = "_None yet._"

    # Improvement stats
    if baseline and best_val_loss is not None and baseline["val_loss"] is not None:
        improvement = (baseline["val_loss"] - best_val_loss) / baseline["val_loss"] * 100
        improvement_str = f"-{improvement:.1f}%"
        baseline_str = f"{baseline['val_loss']:.6f}"
    else:
        improvement_str = "N/A"
        baseline_str = "N/A"

    best_str = f"{best_val_loss:.6f}" if best_val_loss is not None else "N/A"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""# Research Context — Updated {timestamp}

## Current Best
{best_section}

## Leaderboard (Top 5)
{leaderboard}
## Recent Non-Improvements (last 3)
{failures_section}
## Progress
- Total experiments: {total}  |  Kept (improvements): {kept_count} ({pct})
- Initial baseline loss: {baseline_str}  |  Best so far: {best_str}  |  Improvement: {improvement_str}
"""

    with open(CONTEXT_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    return content


if __name__ == "__main__":
    generate()
    print(f"CONTEXT.md written to {CONTEXT_PATH}")
