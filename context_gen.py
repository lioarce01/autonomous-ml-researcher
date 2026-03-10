"""
context_gen.py — Reads DB and writes CONTEXT.md for the agent.
"""

import os
from datetime import datetime
import db

CONTEXT_PATH   = os.path.join(os.path.dirname(__file__), "CONTEXT.md")
NOTES_PATH     = os.path.join(os.path.dirname(__file__), "NOTES.md")
MEMORY_LOG_PATH = os.path.join(os.path.dirname(__file__), "MEMORY_LOG.md")

# Ordered list of (display_name, tier, [name_keywords])
# Keywords are checked against all experiment names (lowercase, space-joined).
# NOTE: QK-Norm, Trapezoidal LR, Sliding window SSSL, FA3, Per-param LR are
# already in the v2 baseline — intentionally omitted here.
_TECHNIQUES = [
    # Tier 1 — high impact
    ("Higher LR (QK-Norm enables push)",  1, ["higher_lr", "lr_2e3", "lr_3e3", "lr2e", "lr3e"]),
    ("nGPT (normalized transformer)",     1, ["ngpt", "normalized_gpt"]),
    # Tier 2 — medium
    ("Depth/width tradeoff",  2, ["n_layer_", "n_embd_", "embd_", "deeper", "wider", "layer_"]),
    ("LR tuning",             2, ["lr_", "lr3", "lr6", "learning_rate", "warmdown"]),
    ("Warmup tuning",         2, ["warmup"]),
    ("Batch size",            2, ["batch_", "bs_", "bsz"]),
    ("SWA",                   2, ["swa", "weight_avg", "ema_weights"]),
    ("EMBED_LR_MULT tuning",  2, ["embed_lr", "embdlr", "embed_mult"]),
    ("WINDOW_SIZE tuning",    2, ["window_", "sssl_", "local_attn"]),
    # Tier 3 — lower
    ("Dropout",               3, ["dropout", "drop_"]),
    ("Weight decay",          3, ["weight_decay", "wd_"]),
    ("Block size",            3, ["block_size", "block_", "ctx_", "seq_"]),
    ("AdamW beta2",           3, ["beta2", "b2_"]),
    ("Gradient clipping",     3, ["grad_clip", "clip_", "no_clip"]),
    ("Bias terms",            3, ["_bias"]),
    # Tier 4 — combinations
    ("Bundle A (LR + warmdown)",          4, ["bundle_a", "bundle_lr"]),
    ("Bundle B (all kept combined)",      4, ["bundle_b", "bundle_all"]),
    # Tier 5 — novel / literature-sourced
    ("Peri-LN",               5, ["peri", "peri_ln"]),
    ("Differential attention", 5, ["diff_attn", "differential"]),
    ("MoE MLP",               5, ["moe", "mixture_of_experts"]),
    ("ALiBi",                 5, ["alibi"]),
    ("Novel (agent-proposed)", 5, ["novel_", "_novel"]),
]


def _unexplored_section(all_names):
    name_str = " ".join(all_names).lower()
    by_tier = {}
    for display, tier, keywords in _TECHNIQUES:
        if not any(kw in name_str for kw in keywords):
            by_tier.setdefault(tier, []).append(display)

    if not by_tier:
        return "## Unexplored Techniques\nAll mapped techniques have been attempted.\n"

    tier_labels = {
        1: "Tier 1 (high impact)",
        2: "Tier 2 (medium)",
        3: "Tier 3 (lower)",
        4: "Tier 4 (combinations)",
        5: "Tier 5 (novel / literature)",
    }
    lines = ["## Unexplored Techniques\n"]
    for tier in sorted(by_tier):
        lines.append(f"**{tier_labels[tier]}**: {', '.join(by_tier[tier])}")
    return "\n".join(lines) + "\n"


def _memory_alert_section():
    """Return recent MEMORY_LOG anomalies (non-OK rows) for agent awareness."""
    if not os.path.exists(MEMORY_LOG_PATH):
        return ""
    with open(MEMORY_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Collect anomaly detail lines (start with "> [")
    alerts = [l.rstrip() for l in lines if l.startswith("> [")]
    if not alerts:
        return ""
    recent = alerts[-5:]  # last 5 anomalies
    block = "\n".join(recent)
    return f"\n## Recent Memory Anomalies\n{block}\n"


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
    mem_alerts = _memory_alert_section()

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
{mem_alerts}"""

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
