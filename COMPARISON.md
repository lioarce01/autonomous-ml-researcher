# Our Setup vs. Karpathy's Autoresearch

## Model & Training


| Dimension           | **Ours**                               | **Karpathy's**                                | Winner                                                                            |
| ------------------- | -------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------- |
| Dataset             | TinyStories 10% (~200MB)               | climbmix-400b-shuffle (web text, 400B tokens) | Karpathy (larger, more diverse)                                                   |
| Tokenizer           | BPE, vocab=8192                        | BPE, vocab=8192 (rustbpe+tiktoken)            | Tie                                                                               |
| Metric              | val_bpb                                | val_bpb                                       | Tie                                                                               |
| Optimizer           | Muon + AdamW                           | Muon + AdamW (tuned per param group)          | Karpathy (more granular LR per group: embed/unembed/scalar each have separate LR) |
| LR schedule         | WSD (decay last 10%)                   | Trapezoidal (decay last 50%, decays to 0)     | Unclear — both valid                                                              |
| Logit capping       | `30 * tanh(x/30)`                      | `15 * tanh(x/15)`                             | Tie                                                                               |
| Gradient clipping   | None                                   | None                                          | Tie                                                                               |
| Flash Attention     | `F.scaled_dot_product_attention` (FA2) | Flash Attention 3                             | Karpathy                                                                          |
| QK-Norm             | Not in baseline (in exploration guide) | Yes — in baseline                             | Karpathy                                                                          |
| Sliding window attn | No                                     | Yes — SSSL pattern                            | Karpathy                                                                          |
| Weight tying        | Yes                                    | No                                            | Ours                                                                              |
| RoPE                | Yes                                    | Yes                                           | Tie                                                                               |
| MQA                 | Yes (N_KV_HEAD=1)                      | Yes (configurable)                            | Tie                                                                               |


---

## Agent Loop Architecture


| Dimension                    | **Ours**                                                                          | **Karpathy's**                    | Winner                               |
| ---------------------------- | --------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------ |
| Persistent memory            | `CONTEXT.md` — auto-regenerated with leaderboard, streak, unexplored techniques   | `results.tsv` flat file + git log | **Ours**                             |
| Hypothesis tracking          | Explicit `--hypothesis` field stored in DB, reflected on post-experiment          | Commit message only               | **Ours**                             |
| Experiment DB                | SQLite with `kept`, `hypothesis`, `git_hash`, `timestamp`                         | results.tsv (flat text)           | **Ours**                             |
| Revert mechanism             | DB `kept=1` flag, agent reverts `train.py` manually                               | `git reset HEAD~1`                | Karpathy (atomic, harder to mess up) |
| Failure streak detection     | Automatic — counted from DB, shown in CONTEXT.md, triggers pivot protocol         | None                              | **Ours**                             |
| Exploration guide            | Tiered (1-4) with technique list, auto-tracked in CONTEXT.md "Unexplored" section | None — agent decides freely       | **Ours**                             |
| Literature research protocol | Timeboxed (2 WebSearch + 1 WebFetch), triggered by streak or unfamiliar technique | None formalized                   | **Ours**                             |
| Adaptive budget extension    | Explicit rule: extend to 600s if still converging at 5min                         | None                              | **Ours**                             |
| NOTES.md                     | Agent's persistent notebook, appended verbatim to every CONTEXT.md                | None                              | **Ours**                             |
| Live dashboard               | Streamlit — running best, leaderboard, improvement rate                           | None                              | **Ours**                             |
| Pause/resume                 | `.pause` file checked between experiments                                         | Not formalized                    | **Ours**                             |
| HP sensitivity model         | Tracked in NOTES.md, updated every 5 experiments                                  | None                              | **Ours**                             |


