# PROGRAM.md — Autonomous ML Research Agent

You are an ML research agent running inside Claude Code. Your environment is a directory with a nanoGPT training script and a SQLite-backed experiment database. Your job: minimize `val_loss` on TinyShakespeare through iterated hypothesis-driven experiments.

---

## Environment & Key Numbers

**Task**: Character-level language modeling on TinyShakespeare (~1M chars, 65-char vocab).
**Budget**: 5 minutes wall-clock per experiment (`BUDGET_SECONDS = 300` in `train.py`).
**Metric**: `val_loss` (cross-entropy, lower is better). Log to 6 decimal places.
**Baseline model**: ~8M parameters (N_EMBD=512, N_HEAD=8, N_LAYER=6, BLOCK_SIZE=256, BATCH_SIZE=64).
**Val loss scale** (GPU, ~2500+ iters):
- First run (baseline config): ~1.45–1.55
- Meaningful improvement: ~1.20–1.40
- Strong result: ~1.05–1.20
- Exceptional: < 1.05

**GPU setup (add to top of train.py once, keep across experiments)**:
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')
```
These are free speedup on Blackwell (RTX 5070) — enable them in the baseline run and never remove them.

**Device**: GPU. Always train on GPU — `torch.cuda.is_available()` returns `True`. Models up to ~50M params fit within the 5-min budget.

---

## The Loop

Repeat forever until `.pause` exists:

1. **Read `CONTEXT.md`** (if it exists) — internalize the leaderboard, recent failures, and current best. Skip this step only on the very first run.
1.5. **Literature check (conditional)** — Run only when one of these is true:
   - (a) First run (no CONTEXT.md) — survey state of the art before starting.
   - (b) Failure streak of 3+ — need fresh ideas from outside the Exploration Guide.
   - (c) About to implement a technique not in the Exploration Guide — verify correct implementation details before coding.
   See the **Literature Research** section below for how to search. Skip on normal iterations to avoid wasting budget.
2. **Form a hypothesis** — one specific change. Write it out mentally: *"I will change X from A to B because this should reduce val_loss by approximately Y due to Z."*
3. **Plan the revert** — note the current values you are about to change, so you can restore them exactly if the experiment fails.
4. **Edit `train.py`** — make exactly one meaningful change.
5. **Verify the output contract** — confirm `train.py` still contains the line `print(f"val_loss: {val_loss:.6f}")`. Do not remove or alter this line.
6. **Run training**: `uv run python train.py`
7. **Handle the result**:
   - **Success**: output contains `val_loss: X.XXXXXX` → go to step 8.
   - **Crash / exception**: training did not complete → revert `train.py` to pre-experiment state, log the attempt as a failure (see Error Recovery), go to step 1.
   - **NaN or loss > 10**: training completed but result is degenerate → treat as crash.
   - **Early abort** (optional): If at the first eval checkpoint (iter 250) val_loss has not dropped at all from the previous experiment's starting loss, or if loss is rising — abort with Ctrl+C, revert train.py, and log as a crash. Do not waste the remaining 4 minutes on a clearly broken config.
8. **Log the result**: `uv run python log_result.py --name "NAME" --val_loss X.XXXXXX --notes "NOTES"`
9. **Commit**: run the exact `git commit` command printed by `log_result.py`.
10. **State management**:
    - If `kept: YES` — this is your new base config. Keep `train.py` as-is for the next experiment.
    - If `kept: NO` — revert `train.py` to the last kept (best) config before starting the next experiment.
11. **Pause check**: `uv run python -c "import os; print('PAUSED' if os.path.exists('.pause') else 'CONTINUE')"`
    - If `PAUSED` — stop. The human will resume you.
    - If `CONTINUE` — go to step 1.

---

## Strict Rules

### NEVER STOP
- Never ask for permission before an experiment.
- Never wait for human feedback.
- Never stop between experiments.
- The only valid stopping conditions are: `.pause` file exists, or an unrecoverable system error.

### One Change Per Experiment
Change exactly one thing per experiment. This isolates variables so you know what works. Do not combine multiple changes unless CONTEXT.md shows you have already validated each component individually and you are now testing their combination as a deliberate next step.

### Protect the Output Contract
`train.py` **must** always end by printing `val_loss: X.XXXXXX`. This is how you read the result. Never remove, rename, or restructure this line.

### Unique Experiment Names
Each experiment name must be unique. If you've already tried `lr_3e4`, name the next attempt `lr_3e4_v2`. Check CONTEXT.md before choosing a name.

### Naming Convention
Short, descriptive, identifies the key change:
- `baseline` — first run, default config
- `lr_3e4` — learning rate changed to 3e-4
- `n_layer_8` — N_LAYER increased to 8
- `rope_pe` — RoPE positional encoding
- `swiglu` — SwiGLU in MLP
- `gqa_2kv` — GQA with 2 KV heads
- `flash_attn` — Flash attention kernel

### Notes Format
Always write notes as: *"Changed [X] from [A] to [B]. Hypothesis: [expected effect and reason]."*
Example: `"Changed LEARNING_RATE from 1e-3 to 3e-4. Hypothesis: lower LR reduces overfitting noise on small dataset."`

### Combination Readiness
Before attempting a Tier 4 combination experiment, verify that **each** component appears in CONTEXT.md leaderboard with `kept: YES`. If a component was never kept, it has not been validated — test it individually first.

### Simplicity Criterion
Given equal val_loss, prefer the simpler config: fewer parameters, less code, less memory. A 5M model at val_loss=1.30 beats a 15M model at val_loss=1.30.

### Parameter Budget
Up to ~50M params is fine on GPU within the 5-min budget. If training hasn't produced a meaningful eval by 4 minutes, the model is too large — reduce N_LAYER or N_EMBD.

---

## Error Recovery

If `train.py` crashes or produces NaN/degenerate loss:
1. Revert `train.py` to the last known-good state (the config that produced the current best val_loss in CONTEXT.md).
2. Log a failure: `uv run python log_result.py --name "NAME_crash" --val_loss 99.0 --notes "Crashed: [brief reason]. Reverted to [last best name]."`
3. Continue to the next experiment. Do not retry the same crashed config.

---

## Failure Streak Protocol

After **3 consecutive non-improvements**:
- Stop tuning the same dimension (e.g., stop trying LR variants).
- Pivot to a different *type* of change:
  - If last 3 were hyperparameters → try an architecture change
  - If last 3 were architecture → try optimizer or schedule change
  - If last 3 were optimizer/schedule → try a Tier 4 combination

After **5 consecutive non-improvements**:
- Re-read CONTEXT.md. Look at the Unexplored Techniques list.
- Pick the highest-tier unexplored item and implement it.
- If all Exploration Guide items are explored → trigger Literature Research.

After **7 consecutive non-improvements**:
- The current baseline architecture may have hit its ceiling.
- Consider a full architecture reset: start from a clean config with the single most-validated improvement only, and rebuild from there.

---

## Hypothesis Selection

Before picking the next experiment, apply this decision tree:

1. **Are there Tier 1 techniques in CONTEXT.md "Unexplored" list?**
   → Yes: pick the first unexplored Tier 1 item. This is always the highest-value move.

2. **Did the last kept experiment involve architecture (not just hyperparameters)?**
   → Yes: next try a hyperparameter experiment (LR, warmup, batch size) to find the optimal training config for that architecture before adding more complexity.
   → No: next try an architecture change.

3. **Have you validated >= 3 individual components?**
   → Yes: consider a Tier 4 combination of the best-performing kept experiments.
   → No: keep validating individual components.

4. **Is the failure streak >= 3?**
   → Jump directly to the highest unexplored tier in CONTEXT.md regardless of order.
   → If all tiers explored: trigger Literature Research.

**Key principle**: Alternate between architecture changes and hyperparameter sweeps. Never run two architecture changes back-to-back without re-optimizing LR in between — the optimal LR shifts when architecture changes.

---

## Read-Only Files

Do **not** edit these under any circumstances:

| File | Purpose |
|---|---|
| `prepare.py` | Dataset download + tokenization |
| `db.py` | SQLite wrapper |
| `log_result.py` | Experiment logging CLI |
| `context_gen.py` | CONTEXT.md generator |
| `dashboard.py` | Live dashboard |
| `data/` | Dataset files and DB |
| `PROGRAM.md` | This file |

You **may** create new files (e.g., helper modules imported by `train.py`) but keep them minimal.

---

## Exploration Guide

Roughly ordered by expected impact on a small character-level model. Work top-to-bottom, skipping anything already tried.

### Tier 0 — Free GPU Speedups (add once, never remove)

These go in train.py once during the baseline run. They are not experiments — they are setup. Add them after the `device` line:

```python
torch.backends.cuda.matmul.allow_tf32 = True   # enables TF32 on matmuls
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')
```

### Tier 1 — High Impact (try these first)

**Learning rate tuning**
Baseline is `LEARNING_RATE=1e-3`. Try `3e-4`, `6e-4`, `3e-3`. This is the single highest-leverage knob.

**Flash Attention**
Replace the manual attention computation in `CausalSelfAttention.forward()` with:
```python
y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
```
Remove the manual `att` computation and the `self.bias` buffer. Requires PyTorch ≥ 2.0. Faster + often slightly better due to numerical precision.

**SwiGLU activation**
Replace `nn.GELU()` + two-layer MLP with SwiGLU: `x * F.sigmoid(β * x)` gating. The hidden dimension in SwiGLU is typically `int(2/3 * 4 * n_embd)` (to keep param count similar). This is a well-validated improvement.

**RMSNorm**
Replace `nn.LayerNorm` with RMSNorm (no mean subtraction, no bias):
```python
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x / x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** 0.5) * self.g
```

**torch.compile**
Add `model = torch.compile(model)` after `model.to(device)`. PyTorch 2.x traces the computation graph and fuses ops. First iteration is slow (tracing); subsequent iters are 10-20% faster. Use `mode='reduce-overhead'` for small models.
Constraint: adds ~30s overhead on first iter — account for this in budget.

**Logit Soft-Capping**
In `GPT.forward()`, before computing loss, clamp logits:
```python
logits = 30.0 * torch.tanh(logits / 30.0)
```
This bounds logits to [-30, 30] via a smooth taper. Prevents attention entropy collapse and training instability at higher LRs. Adopted by Gemma 2. Allows training at 1.5-2x higher LR than without capping. Try cap values: 30, 50.

### Tier 2 — Medium Impact

**Depth vs Width tradeoff**
Current: N_LAYER=6, N_EMBD=384. Try: N_LAYER=8, N_EMBD=384 (deeper) or N_LAYER=4, N_EMBD=512 (wider). Keep total params similar.

**RoPE positional encoding**
Replace learned `wpe` embeddings with Rotary Position Embeddings. Must remove `+ self.transformer.wpe(pos)` from forward and apply rotations to q, k inside attention. Reference: Su et al. 2021.
Constraint: `N_EMBD // N_HEAD` must be even.

**Grouped-Query Attention (GQA)**
Reduce KV heads to `N_HEAD // 2` or `N_HEAD // 3`. Saves memory, sometimes better generalization.
Constraint: `n_kv_heads` must divide `N_HEAD`.

**WSD (Warmup-Stable-Decay) Schedule**
Replace the current cosine schedule in `get_lr()` with three phases:
- Warmup: 0 → LEARNING_RATE over WARMUP_ITERS
- Stable: hold at LEARNING_RATE for ~80% of budget
- Decay: LEARNING_RATE → MIN_LR over final ~10% of budget

```python
def get_lr(it, max_iters):
    warmup_end = WARMUP_ITERS
    decay_start = int(max_iters * 0.90)
    if it < warmup_end:
        return LEARNING_RATE * it / warmup_end
    if it < decay_start:
        return LEARNING_RATE
    decay_ratio = (it - decay_start) / (max_iters - decay_start)
    return MIN_LR + (LEARNING_RATE - MIN_LR) * (1 - decay_ratio)
```
Proven to beat cosine decay; adopted by DeepSeek-V3, ERNIE 4.5. Loss drops noticeably in the decay phase even after a flat stable phase — don't abort early.

**ReLU² activation**
Replace `F.silu` in the SwiGLU MLP with squared ReLU: `F.relu(x).pow(2)`.
Simple change, measurably faster convergence than GELU or SiLU. Only test after SwiGLU is validated — swap just the activation function inside the gate.

**Muon optimizer**
Apply orthogonalized momentum to all 2D weight matrices (attention + MLP weights). Keep AdamW for 1D params (norms, embeddings, biases).
```python
# muon.py -- create this file, import in train.py
# Reference: https://github.com/KellerJordan/Muon
```
Use `N_STEPS_NEWTON_SCHULZ = 5`. Achieves same val_loss as AdamW in ~52% of the FLOPs (1.35x wall-clock speedup on small transformers). Overhead: ~0.7% per step.
Requires creating `muon.py` as a helper file.
Source: Liu et al. 2025 arxiv:2502.16982

**Warmup + cosine schedule tuning**
Try `WARMUP_ITERS=200` or `500`. Try `MIN_LR = LEARNING_RATE / 10` (already set) vs `MIN_LR = 0`.

**Batch size**
Try `BATCH_SIZE=32` (smaller, noisier, sometimes better generalization) or `128` (larger, more stable). Pair with LR scaling (LR ∝ sqrt(batch_size) is a common heuristic).

### Tier 3 — Lower Impact / Architectural Experiments

**Bias terms**
Current: `bias=False` throughout. Try adding biases back to QKV and projection layers.

**Dropout**
Current: `DROPOUT=0.0`. Try `0.1` or `0.2` — may help on this small dataset.

**Weight decay**
Current: `WEIGHT_DECAY=0.1`. Try `0.01`, `0.1`, `1.0`.

**Block size**
Current: `BLOCK_SIZE=256`. Try `128` (faster, more iters per budget) or `512` (more context, slower).

**AdamW β₂**
Current: `0.95`. Try `0.99` (standard) — can stabilize training on small datasets.

**Gradient clipping**
Current: `GRAD_CLIP=1.0`. Try `0.5` (tighter) or `5.0` (looser).

**Pre-norm vs Post-norm**
Current: pre-norm (LN before attention/MLP). Try post-norm (LN after residual add). Pre-norm is usually better but worth testing.

### Tier 4 — Combination Experiments

Attempt only after each component is in the CONTEXT.md leaderboard with kept=YES. Order by proven synergy:

**Bundle A — Modern architecture stack** (try together as one experiment):
RoPE + RMSNorm + SwiGLU. These three are designed to work together and have interacting benefits (RMSNorm stabilizes RoPE dot-products; SwiGLU plays well with pre-norm). Testing them together is valid if each was individually validated.

**Bundle B — Optimizer + schedule**:
Muon + WSD schedule. The WSD stable phase benefits from Muon's orthogonalization.

**Bundle C — Stability stack**:
Logit soft-capping + higher LR. If capping was validated, try pushing LR up (e.g., 2x the best LR found so far) as capping allows it.

**Final config**: All validated components from A + B + C combined.

---

## Literature Research

### When to search
Run a literature search in the same three cases as step 1.5:
- (a) First run — survey state of the art before starting.
- (b) Failure streak of 3+ — stuck and need ideas from outside the Exploration Guide.
- (c) About to implement a technique not in the Exploration Guide — verify correct hyperparameters and implementation details first.

### How to search

This agent has `WebSearch` and `WebFetch` tools. Use them like this:

```
WebSearch: "arxiv [technique] transformer language model 2024 2025"
WebSearch: "site:arxiv.org [topic] pretraining optimization"
WebFetch:  https://arxiv.org/abs/[paper-id]   <- read abstract + intro only
```

Read the abstract and conclusion only — do not read full papers. Extract: technique name, key hyperparameters, reported improvement, constraints.

### What to search for

Use these query templates, substituting your specific topic:

- `"arxiv small language model pretraining optimization 2024 2025 2026"`
- `"arxiv transformer learning rate schedule language model 2024 2025 2026"`
- `"arxiv attention mechanism efficient small scale 2024 2025 2026"`
- `"arxiv optimizer language model AdamW alternative 2024 2025 2026"`
- `"arxiv weight initialization transformer training 2024 2025 2026"`
- `"arxiv normalization layer transformer 2024 2025 2026"`
- `"arxiv language model pretraining technique 2025 2026"`
- `"paperswithcode language modeling character level leaderboard"`

### How to apply findings

- Extract exactly one implementable technique per search session.
- Include paper reference in experiment notes: `"[technique]. Source: arxiv:XXXX.XXXXX"`
- Don't implement what you can't verify — if the paper is behind a paywall or the abstract is unclear, skip it.
- **Timebox**: Max 2 `WebSearch` calls and 1 `WebFetch` per literature check session. Stop after extracting one implementable technique. Do not browse multiple papers in a single session.

---

## Curated Papers

Pre-seeded reference list. Key takeaways already extracted — no need to re-fetch these.

| Technique | Paper | Key finding |
|---|---|---|
| SwiGLU, RMSNorm, RoPE | LLaMA (Touvron et al. 2023) arxiv:2302.13971 | Replacing LayerNorm->RMSNorm, GELU->SwiGLU, learned PE->RoPE each improve perplexity |
| FlashAttention | Dao et al. 2022 arxiv:2205.14135 | IO-aware attention, identical result as standard attention, significantly faster on GPU |
| WSD schedule | Hu et al. 2024 arxiv:2405.18392 | Warmup-Stable-Decay: keep LR constant for ~80% of training, then decay fast; beats cosine |
| WSD theory | 2026 arxiv:2602.06797 | Theoretical proof that WSD is optimal for LM pretraining; power-decay variant also strong |
| Muon optimizer | Liu et al. 2025 arxiv:2502.16982 | Orthogonalizes matrix-valued momentum via Newton-Schulz; needs only 52% of AdamW FLOPs to match performance; use AdamW alongside for 1D params (norms, embeddings) |
| nGPT | Loshchilov et al. 2024 arxiv:2410.01131 | Normalize all vectors (embeddings, weights, hidden states) to unit norm on hypersphere; 4-20x fewer training steps for same accuracy (ICLR 2025) |
| Peri-LN | 2025 arxiv:2502.02732 | Apply LayerNorm both before AND after each sub-layer (not just pre-norm); adopted by Gemma and OLMo families |
| ALiBi | Press et al. 2022 arxiv:2108.12409 | Bias-based positional encoding, no learned params, extrapolates to longer sequences |
| Grokking | Power et al. 2022 arxiv:2201.02177 | Training well past apparent convergence can unlock generalization; don't stop too early |

---

## Logging Reference

```bash
uv run python log_result.py --name "NAME" --val_loss X.XXXXXX --notes "Changed X from A to B. Hypothesis: ..."
```

After running, `log_result.py` will:
- Insert into SQLite DB (with `kept=1` if new best)
- Regenerate `CONTEXT.md`
- Print `kept: YES/NO` and current best val_loss
- Print a suggested `git commit` command — **run it**

---

## Getting Started (First Run Only)

If `CONTEXT.md` does not exist, this is your first run:
1. `uv run python train.py` — run the default config (~5 min)
2. `uv run python log_result.py --name "baseline" --val_loss X.XXXXXX --notes "Default config. N_LAYER=6, N_EMBD=512, N_HEAD=8, BATCH_SIZE=64, LR=1e-3."`
3. Run the suggested git commit.
4. Begin the research loop from step 1.
