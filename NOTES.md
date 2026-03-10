# Agent Research Notes

This file is written and maintained by the agent. It persists across CONTEXT.md regenerations.
Update it after literature searches, every 5 experiments, and whenever a meaningful insight emerges.

---

## Current Best Config

**Update this section every time a new val_bpb record is set (kept: YES).**

- **Experiment**: `baseline_ext` | **Val BPB**: 0.845044
- N_EMBD=384, N_HEAD=8, N_KV_HEAD=1, N_LAYER=8
- BLOCK_SIZE=256, BATCH_SIZE=128, DROPOUT=0.4, WEIGHT_DECAY=0.1
- LEARNING_RATE=1e-3, MIN_LR=1e-4, WARMUP_ITERS=200, no grad clipping
- Optimizer: Muon (matrix) + AdamW (1D), muon_lr=0.02, muon_momentum=0.95
- MLP: ReGLU (ReLU^2)
- Dataset: TinyStories 10%, BPE vocab=8192
- ~12M params, 1387 iters/10min (680 iters/5min)

---

## HP Sensitivity Map

Update after every 5 experiments. Format: `HP=SENSITIVITY (evidence)`.

Sensitivity unknown on new dataset/tokenizer -- revalidate everything from scratch.

- LR = HIGH (sweet spot ~2e-3; 1e-3→2e-3 delta=0.035 at 5min; 3e-3 overshoots)
- N_EMBD = UNKNOWN (revalidate)
- N_LAYER = UNKNOWN (revalidate)
- DROPOUT = UNKNOWN (0.4 is high; TinyStories larger so may need less -- next to test)
- WEIGHT_DECAY = UNKNOWN (revalidate)
- WARMUP_ITERS = LOW (historically minor effect)

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

5 experiments in. Baseline calibrated: 0.955 at 5min, 0.845 at 10min.

Key insight: the baseline_ext (10min) is hard to beat at 5min because the model genuinely needs ~1400 iters to converge. LR=2e-3 helps at 5min (0.921) but doesn't beat the 10min run. To beat 0.845 at 5min, need either: (a) architectural improvement that lowers asymptotic loss faster, (b) higher LR that converges in fewer iters, or (c) lower dropout that removes regularization overhead.

QK-Norm alone at LR=1e-3 significantly hurt (1.137) -- normalization may conflict with Muon's orthogonalization OR needs higher LR to help (as designed). LR sweep shows 2e-3 is better than both 1e-3 and 3e-3 at 5min. DROPOUT=0.4 is next to investigate -- may be unnecessarily slowing convergence on the larger TinyStories dataset.
