# Our Setup vs. Karpathy's Autoresearch

_Last updated: 2026-03-10 after baseline v2 (FineWeb-Edu + 7 architecture changes)_

## Model & Training

| Dimension           | **Ours**                                               | **Karpathy's**                                | Status |
| ------------------- | ------------------------------------------------------ | --------------------------------------------- | ------ |
| Dataset             | FineWeb-Edu sample-10BT, N_SAMPLES=500k (~400M tokens) | climbmix-400b-shuffle (web text, 400B tokens) | Gap remains (scale only — same dataset class) |
| Tokenizer           | BPE, vocab=8192                                        | BPE, vocab=8192 (rustbpe+tiktoken)            | Tie |
| Metric              | val_bpb                                                | val_bpb                                       | Tie |
| Optimizer           | Muon + AdamW, per-param-group LR (embed/scalar)        | Muon + AdamW (per param group)                | **Closed** |
| LR schedule         | Trapezoidal (50% warmdown, decays to 0)                | Trapezoidal (50% warmdown, decays to 0)       | **Closed** |
| Logit capping       | `15 * tanh(x/15)`                                      | `15 * tanh(x/15)`                             | **Closed** |
| Gradient clipping   | None                                                   | None                                          | Tie |
| Flash Attention     | FA3 conditional (SDPA fallback on Windows/RTX5070)     | Flash Attention 3                             | Tie on H100; minor gap on Windows |
| QK-Norm             | Yes — in baseline                                      | Yes — in baseline                             | **Closed** |
| Sliding window attn | Yes — SSSL (WINDOW_SIZE=64, every 4th layer global)    | Yes — SSSL pattern                            | **Closed** |
| Weight tying        | Yes (saves ~3M params vs vocab_size=8192, n_embd=384)  | No                                            | Ours (debatable) |
| RoPE                | Yes                                                    | Yes                                           | Tie |
| MQA                 | Yes (N_KV_HEAD=1)                                      | Yes (configurable)                            | Tie |
| Memory monitoring   | Yes — VRAM logged per eval, peak at end, anomaly log   | Not formalized                                | Ours |

**Summary**: 5 gaps closed in baseline v2. Remaining gap is dataset scale (500k docs vs 400B tokens) — addressable by increasing N_SAMPLES, same code.

---

## Agent Loop Architecture

| Dimension                    | **Ours**                                                                          | **Karpathy's**                    | Winner |
| ---------------------------- | --------------------------------------------------------------------------------- | --------------------------------- | ------ |
| Persistent memory            | `CONTEXT.md` — auto-regenerated with leaderboard, streak, unexplored techniques   | `results.tsv` flat file + git log | **Ours** |
| Hypothesis tracking          | Explicit `--hypothesis` field stored in DB, reflected on post-experiment          | Commit message only               | **Ours** |
| Experiment DB                | SQLite with `kept`, `hypothesis`, `git_hash`, `timestamp`                         | results.tsv (flat text)           | **Ours** |
| Revert mechanism             | DB `kept=1` flag, agent reverts `train.py` manually                               | `git reset HEAD~1`                | Karpathy (atomic) |
| Failure streak detection     | Automatic — counted from DB, shown in CONTEXT.md, triggers pivot protocol         | None                              | **Ours** |
| Exploration guide            | Tiered (1-5) with technique list, auto-tracked in CONTEXT.md "Unexplored" section | None — agent decides freely       | **Ours** |
| Novel research (Tier 5)      | Explicit free-research mode with [NOVEL] tagging after curated tiers exhausted    | Always free-form                  | Both valid |
| Literature research protocol | Every 5th experiment + on failure/streak (timeboxed 2 search + 1 fetch)           | Not formalized                    | **Ours** |
| Adaptive budget extension    | Explicit rule: extend to 600s if still converging at 5min                         | None                              | **Ours** |
| NOTES.md                     | Agent's persistent notebook, appended verbatim to every CONTEXT.md                | None                              | **Ours** |
| MEMORY_LOG.md                | Append-only VRAM anomaly log ([LEAK]/[FRAG]/[SPIKE]) visible in CONTEXT.md        | None                              | **Ours** |
| Live dashboard               | Streamlit — running best, leaderboard, improvement rate                           | None                              | **Ours** |
| Pause/resume                 | `.pause` file checked between experiments                                         | Not formalized                    | **Ours** |
| HP sensitivity model         | Tracked in NOTES.md, updated every 5 experiments                                  | None                              | **Ours** |
