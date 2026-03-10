# PROGRAM.md — Autonomous ML Research Agent

You are an ML research agent running inside Claude Code. Your environment is a directory with a nanoGPT training script and a SQLite-backed experiment database. Your job: minimize `val_loss` on TinyShakespeare through iterated hypothesis-driven experiments.

---

## Environment & Key Numbers

**Task**: Character-level language modeling on TinyShakespeare (~1M chars, 65-char vocab).
**Budget**: 5 minutes wall-clock per experiment (`BUDGET_SECONDS = 300` in `train.py`).
**Metric**: `val_loss` (cross-entropy, lower is better). Log to 6 decimal places.
**Baseline model**: ~10.6M parameters (N_EMBD=384, N_HEAD=6, N_LAYER=6, BLOCK_SIZE=256).
**Val loss scale**:
- First run (baseline config): ~1.55–1.65
- Meaningful improvement: ~1.30–1.45
- Strong result: ~1.15–1.30
- Exceptional: < 1.15

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
- Stop incremental tuning of the same direction.
- Revert to the current best config (if not already).
- Pivot: try a qualitatively different approach (different architecture component, different optimizer, different data processing).

After **5 consecutive non-improvements**:
- Re-read CONTEXT.md carefully.
- Consider combining previously validated individual improvements.
- Consider undoing all changes and trying the single highest-impact unexplored idea from the exploration list.

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

### Tier 2 — Medium Impact

**Depth vs Width tradeoff**
Current: N_LAYER=6, N_EMBD=384. Try: N_LAYER=8, N_EMBD=384 (deeper) or N_LAYER=4, N_EMBD=512 (wider). Keep total params similar.

**RoPE positional encoding**
Replace learned `wpe` embeddings with Rotary Position Embeddings. Must remove `+ self.transformer.wpe(pos)` from forward and apply rotations to q, k inside attention. Reference: Su et al. 2021.
Constraint: `N_EMBD // N_HEAD` must be even.

**Grouped-Query Attention (GQA)**
Reduce KV heads to `N_HEAD // 2` or `N_HEAD // 3`. Saves memory, sometimes better generalization.
Constraint: `n_kv_heads` must divide `N_HEAD`.

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

### Tier 4 — Combination Experiments (after validating components)

Only attempt these after each component has been individually validated:
- Flash Attention + SwiGLU
- RoPE + RMSNorm
- Best LR + best architecture change
- All validated improvements combined (final config)

---

## Literature Research

### When to search
Run a literature search in the same three cases as step 1.5:
- (a) First run — survey state of the art before starting.
- (b) Failure streak of 3+ — stuck and need ideas from outside the Exploration Guide.
- (c) About to implement a technique not in the Exploration Guide — verify correct hyperparameters and implementation details first.

### How to search

Claude Code has `WebSearch` and `WebFetch` tools. Use them like this:

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
2. `uv run python log_result.py --name "baseline" --val_loss X.XXXXXX --notes "Default nanoGPT config. N_LAYER=6, N_EMBD=384, N_HEAD=6, LR=1e-3."`
3. Run the suggested git commit.
4. Begin the research loop from step 1.
