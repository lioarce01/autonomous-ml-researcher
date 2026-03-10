"""
train.py — nanoGPT on TinyShakespeare with a 5-minute wall-clock budget.

Claude Code is free to edit ANY part of this file to improve val_loss.
After training completes, the script prints: val_loss: X.XXXXXX

Key hyperparameters to tune (at the top for easy access):
"""

import os
import math
import time
import struct

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Hyperparameters — Claude Code edits these (and anything else)
# ---------------------------------------------------------------------------
BUDGET_SECONDS = 300        # wall-clock training budget
BATCH_SIZE     = 128        # micro-batch size
BLOCK_SIZE     = 256        # context length (tokens)
N_EMBD         = 512        # embedding dimension
N_HEAD         = 8          # number of attention heads
N_KV_HEAD      = 1          # number of KV heads for GQA (must divide N_HEAD)
N_LAYER        = 8          # number of transformer blocks
DROPOUT        = 0.4        # dropout (regularize against memorization on GPU)
LEARNING_RATE  = 1e-3       # peak learning rate
MIN_LR         = 1e-4       # minimum LR (cosine decay floor)
WEIGHT_DECAY   = 0.01
GRAD_CLIP      = 0.5
WARMUP_ITERS   = 100        # LR warmup steps
EVAL_INTERVAL  = 250        # evaluate every N iters
EVAL_ITERS     = 50         # batches used for each eval
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_BIN = os.path.join(DATA_DIR, "input.bin")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    if not os.path.exists(INPUT_BIN):
        raise FileNotFoundError(
            f"{INPUT_BIN} not found. Run `python prepare.py` first."
        )
    with open(INPUT_BIN, "rb") as f:
        vocab_size, n_tokens = struct.unpack("<II", f.read(8))
        tokens = list(struct.unpack(f"<{n_tokens}H", f.read(n_tokens * 2)))
    return tokens, vocab_size


def get_batch(data: list[int], batch_size: int, block_size: int):
    import random
    ix = [random.randint(0, len(data) - block_size - 1) for _ in range(batch_size)]
    x = torch.tensor([data[i: i + block_size] for i in ix], dtype=torch.long, device=device)
    y = torch.tensor([data[i + 1: i + block_size + 1] for i in ix], dtype=torch.long, device=device)
    return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_rope(seq_len, head_dim, device):
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    return x * cos.unsqueeze(0).unsqueeze(0) + _rotate_half(x) * sin.unsqueeze(0).unsqueeze(0)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x / x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** 0.5) * self.g


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head
        self.n_embd = n_embd
        self.dropout = dropout
        head_dim = n_embd // n_head
        self.c_q  = nn.Linear(n_embd, n_embd, bias=False)
        self.c_kv = nn.Linear(n_embd, 2 * n_kv_head * head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        head_dim = C // self.n_head
        q = self.c_q(x).view(B, T, self.n_head, head_dim).transpose(1, 2)
        kv = self.c_kv(x)
        k, v = kv.split(self.n_kv_head * head_dim, dim=2)
        k = k.view(B, T, self.n_kv_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, head_dim).transpose(1, 2)
        cos, sin = _build_rope(T, head_dim, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # expand KV to match query heads
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        hidden = int(2 / 3 * 4 * n_embd)  # keep param count same as 2-layer GELU MLP
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up   = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, block_size, dropout):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, block_size, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_kv_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([Block(n_embd, n_head, n_kv_head, block_size, dropout) for _ in range(n_layer)]),
            ln_f=RMSNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        logits = 30.0 * torch.tanh(logits / 30.0)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(it: int, max_iters: int) -> float:
    # WSD: Warmup-Stable-Decay (Hu et al. 2024, arxiv:2405.18392)
    decay_start = int(max_iters * 0.90)
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it < decay_start:
        return LEARNING_RATE
    decay_ratio = (it - decay_start) / max(max_iters - decay_start, 1)
    return MIN_LR + (LEARNING_RATE - MIN_LR) * (1.0 - decay_ratio)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(EVAL_ITERS):
            x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE)
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


# ---------------------------------------------------------------------------
# Muon + AdamW combined optimizer (arxiv:2502.16982)
# ---------------------------------------------------------------------------

class _MuonAdamW:
    """Muon for 2D matrix params, AdamW for 1D (norms, embeddings, biases)."""

    def __init__(self, matrix_params, scalar_params,
                 muon_lr, muon_momentum,
                 adam_lr, adam_betas, adam_wd, ns_steps=5):
        self.muon_lr_base = muon_lr
        self.adam_lr_base = adam_lr
        self.ns_steps = ns_steps
        self.muon_state = {}
        self.muon_params = list(matrix_params)
        for p in self.muon_params:
            self.muon_state[p] = {"buf": torch.zeros_like(p, dtype=torch.float32)}
        self.muon_momentum = muon_momentum
        self.adam = torch.optim.AdamW(scalar_params, lr=adam_lr,
                                      betas=adam_betas, weight_decay=adam_wd)
        self._lr_ratio = 1.0

    def set_lr(self, ratio):
        self._lr_ratio = ratio
        for pg in self.adam.param_groups:
            pg["lr"] = self.adam_lr_base * ratio

    def zero_grad(self, set_to_none=True):
        for p in self.muon_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()
        self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        lr = self.muon_lr_base * self._lr_ratio
        m = self.muon_momentum
        ns = self.ns_steps
        a, b, c = 3.4445, -4.7750, 2.0315
        for p in self.muon_params:
            if p.grad is None:
                continue
            g = p.grad.float()
            # Newton-Schulz orthogonalization
            gv = g.view(g.shape[0], -1)
            norm = gv.norm() + 1e-7
            X = gv / norm
            if X.shape[0] > X.shape[1]:
                X = X.T
            for _ in range(ns):
                A = X @ X.T
                X = a * X + (b * A + c * A @ A) @ X
            if gv.shape[0] > gv.shape[1]:
                X = X.T
            g_orth = (X * norm).view(g.shape)
            # SGD momentum
            state = self.muon_state[p]
            buf = state["buf"]
            buf.mul_(m).add_(g_orth)
            update = g_orth + m * buf  # nesterov
            p.add_(update.to(p.dtype), alpha=-lr)
        self.adam.step()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {device} | dtype: {dtype}")

    tokens, vocab_size = load_data()
    print(f"Dataset: {len(tokens):,} tokens | vocab size: {vocab_size}")

    # Train/val split (90/10)
    n = int(0.9 * len(tokens))
    train_data = tokens[:n]
    val_data = tokens[n:]

    model = GPT(vocab_size, N_EMBD, N_HEAD, N_KV_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT).to(device)
    print(f"Parameters: {model.num_params()/1e6:.2f}M")

    # Optimizer — separate weight decay for tensors >= 2D
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
        betas=(0.9, 0.99),
    )

    t0 = time.time()
    best_val_loss = float("inf")
    iter_num = 0

    print(f"Training for up to {BUDGET_SECONDS}s ...")

    while True:
        elapsed = time.time() - t0
        if elapsed >= BUDGET_SECONDS:
            break

        # Estimate remaining iterations for LR schedule
        approx_iters_per_sec = max(iter_num, 1) / max(elapsed, 1e-6)
        remaining_iters = int((BUDGET_SECONDS - elapsed) * approx_iters_per_sec)
        max_iters = iter_num + remaining_iters + 1

        lr = get_lr(iter_num, max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
        with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            elapsed_str = f"{time.time() - t0:.0f}s"
            print(
                f"  iter {iter_num:5d} | elapsed {elapsed_str:>5} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

        iter_num += 1

    # Final evaluation
    losses = estimate_loss(model, train_data, val_data)
    val_loss = losses["val"]
    total_time = time.time() - t0
    print(f"\nTraining complete: {iter_num} iters in {total_time:.1f}s")

    # This line is read by Claude Code to get the metric
    print(f"val_loss: {val_loss:.6f}")


if __name__ == "__main__":
    main()
