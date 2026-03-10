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

- LR = UNKNOWN (revalidate -- Muon may tolerate higher LR on larger dataset)
- N_EMBD = UNKNOWN (TinyShakespeare sweet spot was 384; TinyStories is larger so bigger models may generalize better)
- N_LAYER = UNKNOWN (revalidate)
- DROPOUT = UNKNOWN (TinyStories is larger -- less overfit risk, may need less dropout)
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

_Fresh start_: Switched from TinyShakespeare (char-level, 65 vocab) to TinyStories (BPE, 8192 vocab).
Dataset is much larger and more diverse -- GPU overfit is less of a problem. Metric is now val_bpb.
Run baseline first to calibrate the bpb scale before drawing conclusions.
