# PROGRAM.md — Autonomous ML Research Agent

You are an ML research agent running inside Claude Code. Your environment is a directory with a nanoGPT training script and a SQLite-backed experiment database. Your job: minimize `val_loss` on TinyShakespeare through iterated hypothesis-driven experiments.

---

## Environment & Key Numbers

**Task**: Character-level language modeling on TinyShakespeare (~1M chars, 65-char vocab).
**Budget**: 5 minutes wall-clock per experiment (`BUDGET_SECONDS = 300` in `train.py`).
**Metric**: `val_loss` (cross-entropy, lower is better). Log to 6 decimal places.
**Baseline model**: ~16M parameters (N_EMBD=512, N_HEAD=8, N_KV_HEAD=1 (MQA), N_LAYER=6,
BLOCK_SIZE=256, BATCH_SIZE=128, DROPOUT=0.4).

**Baseline architecture** (already in train.py — do NOT re-experiment on these):
- RoPE positional encoding (applied to q, k in attention)
- Flash Attention via F.scaled_dot_product_attention
- SwiGLU MLP (gate x up with SiLU, hidden = 2/3 x 4 x n_embd)
- WSD learning rate schedule (warmup-stable-decay)
- MQA (N_KV_HEAD=1)
- bfloat16 autocast + TF32 flags

**Val loss scale** (starting from current best of 1.462):
- Current best: 1.462 (batch128 + dropout_04 validated)
- Meaningful improvement: < 1.43
- Strong result: < 1.30
- Exceptional: < 1.15

**Device**: GPU. Always train on GPU — `torch.cuda.is_available()` returns `True`. Models up to ~50M params fit within the 5-min budget.

**Python environment**: Always use `uv run python` — never bare `python` or `python3`. The project uses a `.venv` managed by uv. Bare `python` resolves to the system interpreter which is missing dependencies (numpy, etc.) and will cause import errors.

If `.venv` does not exist, create it and install all dependencies before doing anything else:
```bash
uv venv .venv
uv pip install -r requirements.txt
```
After that, all commands use `uv run python` as normal.

---

## The Loop

Repeat forever until `.pause` exists:

1. **Read `CONTEXT.md`** (if it exists) — internalize the leaderboard, recent failures, and current best. Skip this step only on the very first run.
   Also verify that train.py hyperparameters match the baseline described in Environment & Key Numbers. If train.py has been left in a modified state from a failed experiment, revert it before proceeding.
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
      Then check for **Adaptive Budget Extension** (see rule below).
    - If `kept: NO` — revert `train.py` to the last kept (best) config before starting the next experiment.
11. **Pause check**: `uv run python -c "import os; print('PAUSED' if os.path.exists('.pause') else 'CONTINUE')"`
    - If `PAUSED` — stop. The human will resume you.
    - If `CONTINUE` — go to step 1.

---

## Adaptive Budget Extension

After a `kept: YES` result, check whether the model was still converging at the end of training.
Look at the printed eval lines: find the last two val_loss values before the final eval.

**Extend if**: val_loss improved by > 0.005 per 100 iters during the last 20% of training.

How to calculate: take the second-to-last eval val and the final val_loss, divide the drop by
the iter gap, scale to per-100-iter rate. Example: iter 750 val=1.493, final (iter 822) val=1.465
→ drop=0.028 over 72 iters → 0.039 per 100 iters → EXTEND (> 0.005 threshold).

**Extension procedure**:
1. Set `BUDGET_SECONDS = 600` in `train.py`.
2. Run training: `uv run python train.py`
3. Log as `[original_name]_ext` with the same notes plus "Extended to 600s — loss still converging."
4. Restore `BUDGET_SECONDS = 300` in `train.py` after logging.
5. If `_ext` is kept → it becomes the new base. If not → the original kept result stays as base.

**Rules**:
- Only extend once per config. If the `_ext` run still shows plateau, accept and move on.
- Never extend a config that was NOT kept.
- Never extend if the last 20% shows < 0.005/100 iter improvement (plateau — more time won't help).

---

## Strict Rules

### NEVER STOP
- Never ask for permission before an experiment.
- Never wait for human feedback.
- Never stop between experiments.
- The only valid stopping conditions are: `.pause` file exists, or an unrecoverable system error.

### Always Use the venv
Every Python command must use `uv run python`. Never use bare `python` or `python3`. The system Python is missing required packages and will fail.

### Do Not Use torch.compile
`torch.compile` requires Triton, which is not available on Windows. Do not add it to `train.py` — it will crash immediately.

### Websearch Before Implementing Unfamiliar Techniques
If you are not fully confident in the exact implementation of a technique (correct formula,
key hyperparameters, code structure), you MUST websearch before writing any code:
1. WebSearch: "[technique name] pytorch implementation"
2. Find a reference implementation or paper abstract
3. Extract the exact implementation details, then code it

Never guess at implementation details. A wrong implementation wastes an entire 5-minute budget
and produces a misleading result in the DB. When in doubt, search first.

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

Note: RoPE, Flash Attention, SwiGLU, WSD schedule, MQA, bfloat16, and TF32 flags are ALL already in train.py baseline. Do NOT experiment on these — they are the starting point.

### Tier 1 — High Impact

**Activate Muon optimizer** [HIGHEST PRIORITY]
The Muon class is already implemented in train.py (see `_MuonAdamW`) but the training
loop uses plain AdamW. To activate it, replace the optimizer setup in `main()`:
```python
matrix_params = [p for p in model.parameters() if p.dim() >= 2]
scalar_params  = [p for p in model.parameters() if p.dim() < 2]
optimizer = _MuonAdamW(
    matrix_params, scalar_params,
    muon_lr=0.02, muon_momentum=0.95,
    adam_lr=LEARNING_RATE, adam_betas=(0.9, 0.99), adam_wd=WEIGHT_DECAY
)
# Update LR in the loop: replace param_group loop with optimizer.set_lr(lr / LEARNING_RATE)
```
Expected: same val_loss as AdamW in ~52% of FLOPs -> more iterations in 5 min.
Source: Liu et al. 2025 arxiv:2502.16982

**RMSNorm**
Replace `nn.LayerNorm` in Block and GPT with RMSNorm (no mean subtraction, no bias):
```python
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x / x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** 0.5) * self.g
```
Replace all `nn.LayerNorm(n_embd)` calls with `RMSNorm(n_embd)`.

**Logit Soft-Capping**
In `GPT.forward()`, before computing loss:
```python
logits = 30.0 * torch.tanh(logits / 30.0)
```
Prevents attention entropy collapse; allows higher LR. Adopted by Gemma 2. Try cap 30 or 50.

### Tier 2 — Medium Impact

**Depth vs Width tradeoff**
Current: N_LAYER=6, N_EMBD=512. Try deeper: N_LAYER=8, N_EMBD=512 (more params, check budget).
Or wider: N_LAYER=4, N_EMBD=640 (more width, fewer layers). Keep total params < 50M.

**LR tuning**
Current: LEARNING_RATE=1e-3. Try 3e-4 (more conservative) or 2e-3 (more aggressive with WSD).
WSD schedule tolerates higher peak LR than cosine — explore the upper end.

**Warmup tuning**
Current: WARMUP_ITERS=100. Try 200 or 50. Affects how quickly LR reaches peak.

**Logit Soft-Capping + higher LR** (after soft-capping validated)
Once capping is validated, try pushing LEARNING_RATE to 2e-3 or 3e-3. Capping prevents
the instability that normally limits LR.

### Tier 3 — Lower Impact

**ReLU2 activation** (only after Muon validated)
In the SwiGLU MLP, replace F.silu with squared ReLU: `F.relu(x).pow(2)`.

**Dropout**
Current: DROPOUT=0.4. Try 0.3 (less regularization) or 0.5 (more).

**Weight decay**
Current: WEIGHT_DECAY=0.01. Try 0.1 (more) or 0.001 (less).

**Block size**
Current: BLOCK_SIZE=256. Try 512 (more context, fewer iters) or 128 (faster, more iters).

**AdamW beta2**
Current: beta2=0.99. Try 0.95 if training is noisy.

**Gradient clipping**
Current: GRAD_CLIP=0.5. Try 1.0 (looser) or 0.25 (tighter).

**Bias terms**
Current: bias=False throughout. Try adding bias=True to QKV and projection layers.

### Tier 4 — Combination Experiments

Attempt only after each component is in the CONTEXT.md leaderboard with kept=YES.

**Bundle A**: Muon + WSD (already have WSD, just activate Muon)
**Bundle B**: RMSNorm + Logit Soft-Capping + higher LR
**Bundle C**: All kept individual improvements combined

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
1. **Check venv**: if `.venv` does not exist, run `uv venv .venv && uv pip install -r requirements.txt` first.
2. `uv run python train.py` — run the default config (~5 min)
3. `uv run python log_result.py --name "baseline" --val_loss X.XXXXXX --notes "Baseline: RoPE+Flash+SwiGLU+MQA+WSD+DROPOUT=0.4+BATCH=128, AdamW LR=1e-3."`
4. Run the suggested git commit.
5. Begin the research loop from step 1.
