# MEMORY_LOG.md — GPU VRAM Monitoring

Append-only log. One row per completed training run. Written by the agent at step 7.5.
Skip rows for CPU runs or crashed runs that produced no "Peak VRAM:" line.

**Status codes**
- `OK`     — nothing unusual
- `[LEAK]` — last_eval_alloc > first_eval_alloc + 0.10 GB (allocations grew across training)
- `[FRAG]` — peak_reserved > peak_alloc * 1.8 (PyTorch holding excessive unused cache)
- `[SPIKE]`— peak_alloc > 2x first_eval_alloc (sudden large allocation mid-run)

**How to read alloc vs reserved**
- `alloc`    = tensors actively referenced (model weights + current activations + optimizer state)
- `reserved` = pages PyTorch is holding in its cache (alloc + fragmentation + pre-allocated pool)
- `reserved >> alloc` = fragmentation or over-caching; rarely a real problem unless OOM is near
- `alloc` growing across evals with no architectural reason = genuine leak (missing `del`, retained graph, etc.)

---

## Log

| Experiment | Peak Alloc | Peak Reserved | First Eval Alloc | Last Eval Alloc | Status |
|---|---|---|---|---|---|
| *(no entries yet — populated after first GPU run)* | | | | | |

---

## Anomaly Details

*(Detail lines for non-OK entries go here. Format: `> [CODE] experiment: observation`)*
