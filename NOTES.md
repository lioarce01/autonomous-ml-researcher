# Agent Research Notes

This file is written and maintained by the agent. It persists across CONTEXT.md regenerations.
Update it after literature searches, every 5 experiments, and whenever a meaningful insight emerges.

---

## Current Best Config

**Update this section every time a new val_bpb record is set (kept: YES).**

- **Experiment**: TBD (new dataset -- run baseline_v2 first)
- N_EMBD=384, N_HEAD=8, N_KV_HEAD=1, N_LAYER=8
- BLOCK_SIZE=256, BATCH_SIZE=128, DROPOUT=0.2, WEIGHT_DECAY=0.1
- LEARNING_RATE=1e-3, MIN_LR=0.0, WARMUP_ITERS=200, WARMDOWN_FRAC=0.5
- Optimizer: Muon (matrix) + AdamW embed (LR*3.0) + AdamW scalar (LR*1.0)
- MLP: ReGLU (ReLU^2)
- Dataset: FineWeb-Edu sample-10BT, N_SAMPLES=500k, BPE vocab=8192
- ~12M params, iter rate TBD on new dataset

---

## HP Sensitivity Map

Update after every 5 experiments. Format: `HP=SENSITIVITY (evidence)`.

**RESET**: New dataset (FineWeb-Edu), new architecture (QK-Norm, SSSL, trap LR, per-param LR).
All sensitivities unknown -- revalidate everything from scratch on baseline_v2.

- LR = UNKNOWN (old sweet spot was ~2e-3 on TinyStories; QK-Norm may enable higher)
- WARMDOWN_FRAC = UNKNOWN (0.5 is untested; 50% decay window may be too aggressive)
- EMBED_LR_MULT = UNKNOWN (3.0 is initial guess; may be too high or too low)
- N_EMBD = UNKNOWN (revalidate)
- N_LAYER = UNKNOWN (revalidate)
- DROPOUT = UNKNOWN (0.2 baseline; FineWeb-Edu is larger so may need less)
- WEIGHT_DECAY = UNKNOWN (revalidate)
- WARMUP_ITERS = LOW (historically minor effect; likely still low)

---

## Research Findings

Add findings from web searches and arxiv here. Include source and key implementation detail.

_No findings logged yet. Add after first literature search._

---

## Implementation Notes

Technique-specific tips discovered during experiments or research.

- **SWA**: init swa_model=None before training loop; deepcopy at 80% mark; evaluate swa_model at end
- **QK-Norm**: apply after q/k linear projection, before RoPE; epsilon 1e-6; may allow 1.5-2x LR
- **Muon**: ACTIVE in baseline. Uses set_lr(ratio) for LR schedule, not param_group loop. muon_lr=0.02 is separate from LEARNING_RATE. If reverting to AdamW for any experiment, restore the param_group loop.
- **val_bpb**: metric = val_loss_nats / (avg_bytes_per_token * ln(2)). avg_bytes_per_token loaded from data/meta.json. BPE vocab=8192 gives ~3-4 bytes/token on English text.

---

## Current Mental Model

Agent's working theory of the loss landscape. Update when the picture changes.

**RESET**: New dataset, new baseline architecture. Previous mental model invalidated.

Starting fresh with FineWeb-Edu (diverse educational web text, ~400M train tokens).
Baseline now includes: QK-Norm, SSSL sliding window, trapezoidal LR, per-param-group LR.
These are all proven in Karpathy speedrun -- the delta vs. old TinyStories baseline is unknown.

FineWeb-Edu is harder than TinyStories (more diverse, longer vocabulary, real-world text).
Expected val_bpb range: 0.8-1.0 (TinyStories was artificially easy ~0.85).
First priority: establish baseline_v2 and calibrate the new loss landscape.
Then: sweep LR (QK-Norm should allow pushing higher), WARMDOWN_FRAC, EMBED_LR_MULT.
