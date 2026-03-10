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
BATCH_SIZE     = 32         # micro-batch size
BLOCK_SIZE     = 128        # context length (tokens)
N_EMBD         = 192        # embedding dimension
N_HEAD         = 6          # number of attention heads
N_LAYER        = 6          # number of transformer blocks
DROPOUT        = 0.0        # dropout (0.0 = off; good for small data)
LEARNING_RATE  = 1e-3       # peak learning rate
MIN_LR         = 1e-4       # minimum LR (cosine decay floor)
WEIGHT_DECAY   = 0.1
GRAD_CLIP      = 1.0
WARMUP_ITERS   = 100        # LR warmup steps
EVAL_INTERVAL  = 250        # evaluate every N iters
EVAL_ITERS     = 50         # batches used for each eval
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_BIN = os.path.join(DATA_DIR, "input.bin")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32


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

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd),
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
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
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
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > max_iters:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (max_iters - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


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
            with torch.autocast(device_type=device, dtype=dtype):
                _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


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

    model = GPT(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT).to(device)
    print(f"Parameters: {model.num_params()/1e6:.2f}M")

    # Optimizer — separate weight decay for tensors ≥ 2D
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
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
        with torch.autocast(device_type=device, dtype=dtype):
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
