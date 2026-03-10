"""
train.py -- nanoGPT on FineWeb-Edu with a 5-minute wall-clock budget.

Claude Code is free to edit ANY part of this file to improve val_bpb.
After training completes, the script prints: val_bpb: X.XXXXXX

Key hyperparameters to tune (at the top for easy access):
"""

import os
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from flash_attn import flash_attn_func
    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Hyperparameters -- Claude Code edits these (and anything else)
# ---------------------------------------------------------------------------
BUDGET_SECONDS     = 300        # wall-clock training budget
BATCH_SIZE         = 128        # micro-batch size
BLOCK_SIZE         = 256        # context length (tokens)
N_EMBD             = 384        # embedding dimension
N_HEAD             = 8          # number of attention heads
N_KV_HEAD          = 1          # number of KV heads for GQA (must divide N_HEAD)
N_LAYER            = 8          # number of transformer blocks
DROPOUT            = 0.2        # dropout (regularize against memorization on GPU)
LEARNING_RATE      = 1e-3       # peak learning rate
MIN_LR             = 0.0        # trapezoidal: decay to 0 (Karpathy style)
WEIGHT_DECAY       = 0.1
WARMUP_ITERS       = 200        # LR warmup steps
WARMDOWN_FRAC      = 0.5        # trapezoidal: fraction of training spent decaying LR
EMBED_LR_MULT      = 3.0        # embeddings LR = LEARNING_RATE * EMBED_LR_MULT
USE_SLIDING_WINDOW = True
WINDOW_SIZE        = 64         # local window for S layers (tokens); every 4th layer is full
EVAL_INTERVAL      = 250        # evaluate every N iters
EVAL_ITERS         = 50         # batches used for each eval
# ---------------------------------------------------------------------------

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
INPUT_BIN = os.path.join(DATA_DIR, "input.bin")
VAL_BIN   = os.path.join(DATA_DIR, "val.bin")
META_JSON = os.path.join(DATA_DIR, "meta.json")

HEADER_SIZE = 16  # bytes: magic(4) + version(4) + vocab_size(4) + n_tokens(4)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


# ---------------------------------------------------------------------------
# Data loading -- numpy memmap (O(1) memory regardless of dataset size)
# ---------------------------------------------------------------------------

def load_data():
    for path in (INPUT_BIN, VAL_BIN, META_JSON):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run `python prepare.py` first.")

    with open(META_JSON) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]
    avg_bytes_per_token = meta["avg_bytes_per_token"]

    train_data = np.memmap(INPUT_BIN, dtype=np.uint16, mode='r', offset=HEADER_SIZE)
    val_data   = np.memmap(VAL_BIN,   dtype=np.uint16, mode='r', offset=HEADER_SIZE)

    return train_data, val_data, vocab_size, avg_bytes_per_token


def get_batch(data: np.ndarray, batch_size: int, block_size: int):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i.item():i.item()+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i.item()+1:i.item()+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


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
    def __init__(self, n_embd, n_head, n_kv_head, block_size, dropout, window_size=None):
        super().__init__()
        assert n_embd % n_head == 0
        assert n_head % n_kv_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.window_size = window_size
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
        # QK-Norm: normalize q and k before RoPE (stabilizes attention, enables higher LR)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        cos, sin = _build_rope(T, head_dim, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # expand KV to match query heads (GQA -> MHA)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        if _FLASH_AVAILABLE and self.window_size is None:
            # Flash Attention 3: expects (B, T, H, D)
            q_fa = q.transpose(1, 2)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            y = flash_attn_func(q_fa, k_fa, v_fa,
                                dropout_p=self.dropout if self.training else 0,
                                causal=True)
            y = y.reshape(B, T, C)
        elif self.window_size is not None:
            # Sliding window causal mask (S layers in SSSL)
            rows = torch.arange(T, device=x.device).unsqueeze(1)
            cols = torch.arange(T, device=x.device).unsqueeze(0)
            mask = (cols <= rows) & (cols > rows - self.window_size)
            additive = torch.full((T, T), float('-inf'), device=x.device, dtype=torch.float32)
            additive = additive.masked_fill(mask, 0.0)
            y = F.scaled_dot_product_attention(q, k, v,
                    attn_mask=additive.unsqueeze(0).unsqueeze(0),
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
        else:
            # Standard SDPA (full causal attention, SDPA fallback)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                    dropout_p=self.dropout if self.training else 0, is_causal=True)
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
        return self.drop(self.down(F.relu(self.gate(x)).pow(2) * self.up(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, block_size, dropout, window_size=None):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, block_size, dropout,
                                        window_size=window_size)
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

        def _window_for_layer(i):
            # SSSL: every 4th layer (i % 4 == 3) is full attention; rest are local window
            if not USE_SLIDING_WINDOW:
                return None
            return None if (i % 4 == 3) else WINDOW_SIZE

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([
                Block(n_embd, n_head, n_kv_head, block_size, dropout,
                      window_size=_window_for_layer(i))
                for i in range(n_layer)
            ]),
            ln_f=RMSNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying: wte and lm_head share the same tensor
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
        logits = 15.0 * torch.tanh(logits / 15.0)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# LR schedule -- Trapezoidal (warmup -> hold -> linear decay to MIN_LR)
# ---------------------------------------------------------------------------

def get_lr(it: int, max_iters: int) -> float:
    # Trapezoidal: last WARMDOWN_FRAC of training linearly decays to MIN_LR
    warmdown_start = int(max_iters * (1.0 - WARMDOWN_FRAC))
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / max(WARMUP_ITERS, 1)
    if it < warmdown_start:
        return LEARNING_RATE
    decay_ratio = (it - warmdown_start) / max(max_iters - warmdown_start, 1)
    return LEARNING_RATE * (1.0 - decay_ratio) + MIN_LR * decay_ratio


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
    """Muon for 2D matrix params, AdamW for embeddings (high LR) and 1D scalars."""

    def __init__(self, matrix_params, embed_params, scalar_params,
                 muon_lr, muon_momentum,
                 adam_lr, embed_lr, adam_betas, adam_wd, ns_steps=5):
        self.muon_lr_base = muon_lr
        self.ns_steps = ns_steps
        self.muon_state = {}
        self.muon_params = list(matrix_params)
        for p in self.muon_params:
            self.muon_state[p] = {"buf": torch.zeros_like(p, dtype=torch.float32)}
        self.muon_momentum = muon_momentum
        # Per-param-group LR: embeddings get higher LR, scalars get standard LR
        self.adam = torch.optim.AdamW([
            {"params": list(embed_params),  "lr": embed_lr,  "_base_lr": embed_lr},
            {"params": list(scalar_params), "lr": adam_lr,   "_base_lr": adam_lr},
        ], betas=adam_betas, weight_decay=adam_wd)
        self._lr_ratio = 1.0

    def set_lr(self, ratio):
        self._lr_ratio = ratio
        for pg in self.adam.param_groups:
            pg["lr"] = pg["_base_lr"] * ratio

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
# Memory monitoring
# ---------------------------------------------------------------------------

def _mem_str() -> str:
    """GPU VRAM usage string: 'mem X.XGB/Y.YGB (alloc/reserved)'. Empty string on CPU."""
    if device != "cuda":
        return ""
    alloc    = torch.cuda.memory_allocated()  / 1024**3
    reserved = torch.cuda.memory_reserved()   / 1024**3
    return f" | mem {alloc:.2f}/{reserved:.2f}GB"


def _print_mem_summary() -> None:
    """Print peak GPU VRAM stats at end of training. No-op on CPU."""
    if device != "cuda":
        return
    peak_alloc    = torch.cuda.max_memory_allocated()  / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved()   / 1024**3
    print(f"Peak VRAM: {peak_alloc:.2f}GB allocated | {peak_reserved:.2f}GB reserved")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {device} | dtype: {dtype} | Flash Attention: {_FLASH_AVAILABLE}")

    train_data, val_data, vocab_size, avg_bytes_per_token = load_data()
    print(f"Dataset: {len(train_data):,} train tokens | {len(val_data):,} val tokens | vocab size: {vocab_size}")

    model = GPT(vocab_size, N_EMBD, N_HEAD, N_KV_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT).to(device)
    print(f"Parameters: {model.num_params()/1e6:.2f}M{_mem_str()}")

    # Optimizer: Muon for matrix params, AdamW (high LR) for embeddings, AdamW for scalars
    # Embeddings share weight with lm_head (weight tying) -- treat as embed group (high LR)
    embed_params  = [model.transformer.wte.weight]  # also == lm_head.weight (tied)
    matrix_params = [p for n, p in model.named_parameters()
                     if p.dim() >= 2 and p is not model.transformer.wte.weight]
    scalar_params  = [p for p in model.parameters() if p.dim() < 2]
    optimizer = _MuonAdamW(
        matrix_params, embed_params, scalar_params,
        muon_lr=0.02, muon_momentum=0.95,
        adam_lr=LEARNING_RATE,
        embed_lr=LEARNING_RATE * EMBED_LR_MULT,
        adam_betas=(0.9, 0.99), adam_wd=WEIGHT_DECAY,
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
        optimizer.set_lr(lr / LEARNING_RATE)

        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
        with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            val_bpb_mid = losses["val"] / (avg_bytes_per_token * math.log(2))
            elapsed_str = f"{time.time() - t0:.0f}s"
            print(
                f"  iter {iter_num:5d} | elapsed {elapsed_str:>5} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"val_bpb {val_bpb_mid:.4f} | lr {lr:.2e}{_mem_str()}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

        iter_num += 1

    # Final evaluation
    losses = estimate_loss(model, train_data, val_data)
    val_loss = losses["val"]
    val_bpb = val_loss / (avg_bytes_per_token * math.log(2))
    total_time = time.time() - t0
    print(f"\nTraining complete: {iter_num} iters in {total_time:.1f}s")
    _print_mem_summary()

    # This line is read by Claude Code to get the metric
    print(f"val_bpb: {val_bpb:.6f}")


if __name__ == "__main__":
    main()
