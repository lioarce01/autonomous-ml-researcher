# Agent Research Notes

This file is written and maintained by the agent. It persists across CONTEXT.md regenerations.
Update it after literature searches, every 5 experiments, and whenever a meaningful insight emerges.

---

## Current Best Config

**Update this section every time a new val_loss record is set (kept: YES).**

- **Experiment**: `embd_384` | **Val Loss**: `1.459278`
- N_EMBD=384, N_HEAD=8, N_KV_HEAD=1, N_LAYER=8
- BLOCK_SIZE=256, BATCH_SIZE=128, DROPOUT=0.4, WEIGHT_DECAY=0.1
- LEARNING_RATE=1e-3, MIN_LR=1e-4, WARMUP_ITERS=200, GRAD_CLIP=0.5
- ~12M params

---

## HP Sensitivity Map

Update after every 5 experiments. Format: `HP=SENSITIVITY (evidence)`.

- LR = HIGH (lr_1e5_embd384 +1.5x LR plateaued; 1e-3 clearly better for embd_384)
- N_EMBD = HIGH (embd_384 beat 512 by 0.004 — smaller model generalizes better on 1M chars)
- N_LAYER = HIGH (n_layer_8 improved over 6; but n_layer_10 not yet validated)
- DROPOUT = MEDIUM (dropout_03 overfit; 0.4 is the validated sweet spot so far)
- WEIGHT_DECAY = MEDIUM (wd_01 improved; 0.1 > 0.01 on this dataset)
- WARMUP_ITERS = LOW (warmup_200 minor improvement; not a primary lever)
- BLOCK_SIZE = LOW-MEDIUM (block_128 hurt; 256 is good; 512 untested)

---

## Research Findings

Add findings from web searches and arxiv here. Include source and key implementation detail.

_No findings logged yet. Add after first literature search._

---

## Implementation Notes

Technique-specific tips discovered during experiments or research.

- **SWA**: init swa_model=None before training loop; deepcopy at 80% mark; evaluate swa_model at end
- **QK-Norm**: apply after q/k linear projection, before RoPE; epsilon 1e-6; may allow 1.5-2x LR
- **Muon**: already in train.py as _MuonAdamW; just swap optimizer setup; use set_lr() in loop

---

## Current Mental Model

Agent's working theory of the loss landscape. Update when the picture changes.

_Current architecture sweet spot_: ~12M params (N_EMBD=384, N_LAYER=8-10) outperforms larger
models on TinyShakespeare (~1M chars). GPU speed causes overfit on big models — regularization
(DROPOUT=0.4, WD=0.1) is essential. LR=1e-3 with WSD is validated; higher LR (1.5e-3+) plateaus.

_Next priority_: Muon optimizer (already implemented in train.py, just needs activation) or
QK-Norm (stabilizes attention, potentially unlocks higher LR).
