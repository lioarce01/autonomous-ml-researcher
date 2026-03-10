"""
context_gen.py — Reads DB and writes CONTEXT.md for the agent.
"""

import os
from datetime import datetime
import db

CONTEXT_PATH = os.path.join(os.path.dirname(__file__), "CONTEXT.md")
NOTES_PATH   = os.path.join(os.path.dirname(__file__), "NOTES.md")

# Ordered list of (display_name, tier, [name_keywords])
# Keywords are checked against all experiment names (lowercase, space-joined)
_TECHNIQUES = [
    # Tier 1 — high impact (Flash Attention, SwiGLU already in baseline — omitted)
    ("Learning rate tuning",  1, ["lr_", "lr3", "lr6", "learning_rate"]),
    ("QK-Norm",               1, ["qk_norm", "qknorm"]),
    ("nGPT",                  1, ["ngpt", "normalized_gpt"]),
    # Tier 2 — medium (RoPE, GQA/MQA, WSD already in baseline — omitted)
    ("Depth/width tradeoff",  2, ["n_layer_", "n_embd_", "embd_", "deeper", "wider", "layer_"]),
    ("Warmup tuning",         2, ["warmup"]),
    ("Batch size",            2, ["batch_", "bs_", "bsz"]),
    ("SWA",                   2, ["swa", "weight_avg", "ema_weights"]),
    ("Trapezoidal LR",        2, ["trapezoid", "trapezoidal"]),
    ("Higher LR",             2, ["higher_lr", "lr_2e3", "lr_3e3"]),
    # Tier 3 — lower
    ("Bias terms",            3, ["_bias"]),
    ("Dropout",               3, ["dropout", "drop_"]),
    ("Weight decay",          3, ["weight_decay", "wd_"]),
    ("Block size",            3, ["block_size", "block_", "ctx_", "seq_"]),
    ("AdamW beta2",           3, ["beta2", "b2_"]),
    ("Gradient clipping",     3, ["grad_clip", "clip_", "no_clip", "clip_off", "clip_none", "clip_1", "clip_5"]),
    ("Post-norm",             3, ["post_norm", "postnorm"]),
    # Tier 4 — literature / ambitious
    ("Peri-LN",               4, ["peri", "peri_ln"]),
]


def _unexplored_section(all_names):
    name_str = " ".join(all_names).lower()
    by_tier = {}
    for display, tier, keywords in _TECHNIQUES:
        if not any(kw in name_str for kw in keywords):
            by_tier.setdefault(tier, []).append(display)

    if not by_tier:
        return "## Unexplored Techniques\nAll mapped techniques have been attempted.\n"

    tier_labels = {1: "Tier 1 (high impact)", 2: "Tier 2 (medium)", 3: "Tier 3 (lower)", 4: "Tier 4 / Literature"}
    lines = ["## Unexplored Techniques\n"]
    for tier in sorted(by_tier):
        lines.append(f"**{tier_labels[tier]}**: {', '.join(by_tier[tier])}")
    return "\n".join(lines) + "\n"


def generate():
    stats = db.get_stats()
    top = db.get_top(5)
    failures = db.get_recent_failures(3)
    baseline = db.get_baseline()
    streak = db.get_failure_streak()
    all_names = [e["name"] for e in db.get_all()]

    total = stats["total"] or 0
    kept_count = int(stats["kept_count"] or 0)
    best_val_bpb = stats["best_val_bpb"]
    pct = f"{kept_count/total*100:.0f}%" if total > 0 else "N/A"

    # Current best
    if top:
        best = top[0]
        streak_str = f"  |  Failure streak: {streak}" if streak > 0 else ""
        best_section = (
            f"**Experiment**: `{best['name']}` | "
            f"**Val BPB**: `{best['val_bpb']:.6f}` | "
            f"**Notes**: {best['notes'] or '-'}"
            f"{streak_str}"
        )
    else:
        best_section = "_No experiments logged yet._"

    # Leaderboard table
    if top:
        header = "| Rank | Name | Val BPB | Notes | When |\n|---|---|---|---|---|\n"
        rows = ""
        for i, exp in enumerate(top, 1):
            rows += (
                f"| {i} | `{exp['name']}` | `{exp['val_bpb']:.6f}` "
                f"| {exp['notes'] or '-'} | {exp['timestamp'][:16]} |\n"
            )
        leaderboard = header + rows
    else:
        leaderboard = "_No experiments yet._"

    # Recent non-improvements
    if failures:
        f_header = "| Name | Val BPB | Notes |\n|---|---|---|\n"
        f_rows = ""
        for exp in failures:
            f_rows += (
                f"| `{exp['name']}` | `{exp['val_bpb']:.6f}` "
                f"| {exp['notes'] or '-'} |\n"
            )
        failures_section = f_header + f_rows
    else:
        failures_section = "_None yet._"

    # Improvement stats
    if baseline and best_val_bpb is not None and baseline["val_bpb"] is not None:
        improvement = (baseline["val_bpb"] - best_val_bpb) / baseline["val_bpb"] * 100
        improvement_str = f"-{improvement:.1f}%"
        baseline_str = f"{baseline['val_bpb']:.6f}"
    else:
        improvement_str = "N/A"
        baseline_str = "N/A"

    best_str = f"{best_val_bpb:.6f}" if best_val_bpb is not None else "N/A"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    unexplored = _unexplored_section(all_names)

    content = f"""# Research Context -- Updated {timestamp}

## Current Best
{best_section}

## Leaderboard (Top 5)
{leaderboard}
## Recent Non-Improvements (last 3)
{failures_section}
{unexplored}
## Progress
- Total experiments: {total}  |  Kept (improvements): {kept_count} ({pct})
- Initial baseline bpb: {baseline_str}  |  Best so far: {best_str}  |  Improvement: {improvement_str}
"""

    # Append agent-written notes if they exist (survives regeneration)
    if os.path.exists(NOTES_PATH):
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            notes = f.read().strip()
        if notes:
            content += f"\n---\n\n{notes}\n"

    with open(CONTEXT_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    return content


if __name__ == "__main__":
    generate()
    print(f"CONTEXT.md written to {CONTEXT_PATH}")
