"""
prepare.py -- Downloads FineWeb-Edu, trains BPE tokenizer (vocab=8192), tokenizes to binary.
Run once before training. Outputs:
  data/input.bin      -- uint16 tokens (train split), 16-byte header
  data/val.bin        -- uint16 tokens (val split), 16-byte header
  data/meta.json      -- vocab_size, avg_bytes_per_token (needed for val_bpb)
  data/tokenizer.json -- saved BPE tokenizer (for inspection/reuse)

Binary format (both input.bin and val.bin):
  Header (16 bytes): magic(uint32) version(uint32) vocab_size(uint32) n_tokens(uint32)
  Body: n_tokens x uint16 tokens

Scale knobs:
  N_SAMPLES = 500_000 for RTX 5070 (~400M tokens)
  N_SAMPLES = 5_000_000 for H100  (~4B tokens)
"""
import os, json, struct
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
VOCAB_SIZE = 8192
DATASET    = "HuggingFaceFW/fineweb-edu"
SUBSET     = "sample-10BT"
N_SAMPLES  = 500_000   # train docs (~400M tokens on RTX 5070); increase for H100
N_VAL      = 10_000    # val docs (~8M tokens, enough for stable eval)

MAGIC   = 0x544F4B53   # "TOKS"
VERSION = 1


def write_bin(path, tokens, vocab_size):
    """Write tokens to binary file with 16-byte header."""
    arr = np.array(tokens, dtype=np.uint16)
    n = len(arr)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", MAGIC, VERSION, vocab_size, n))
        arr.tofile(f)
    return n


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Load FineWeb-Edu -- take only what we need (no full 10BT download)
    total = N_SAMPLES + N_VAL
    print(f"Loading {DATASET} ({SUBSET}), taking {total:,} docs...")
    ds = load_dataset(DATASET, name=SUBSET, split=f"train[:{total}]")
    train_ds = ds.select(range(N_SAMPLES))
    val_ds   = ds.select(range(N_SAMPLES, len(ds)))
    train_texts = train_ds["text"]
    val_texts   = val_ds["text"]
    print(f"  Train docs: {len(train_texts):,} | Val docs: {len(val_texts):,}")

    # 2. Train BPE tokenizer on train text
    print(f"Training BPE tokenizer (vocab={VOCAB_SIZE})...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]"])
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    tokenizer.save(os.path.join(DATA_DIR, "tokenizer.json"))
    print(f"  Tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")

    # 3. Tokenize train split and compute avg_bytes_per_token
    print("Tokenizing train split...")
    all_tokens = []
    total_bytes = 0
    for i, text in enumerate(train_texts):
        enc = tokenizer.encode(text)
        all_tokens.extend(enc.ids)
        total_bytes += len(text.encode("utf-8"))
        if (i + 1) % 50_000 == 0:
            print(f"  {i+1:,}/{len(train_texts):,} docs, {len(all_tokens):,} tokens")
    avg_bytes_per_token = total_bytes / len(all_tokens)

    # Write train binary with 16-byte header
    n_train = write_bin(os.path.join(DATA_DIR, "input.bin"),
                        all_tokens, tokenizer.get_vocab_size())

    # 4. Tokenize val split
    print("Tokenizing val split...")
    val_tokens = []
    for text in val_texts:
        val_tokens.extend(tokenizer.encode(text).ids)
    n_val = write_bin(os.path.join(DATA_DIR, "val.bin"),
                      val_tokens, tokenizer.get_vocab_size())

    # 5. Write meta.json
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "avg_bytes_per_token": avg_bytes_per_token,
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone.")
    print(f"  Train tokens : {n_train:,}")
    print(f"  Val tokens   : {n_val:,}")
    print(f"  avg bytes/tok: {avg_bytes_per_token:.3f}")
    print(f"  Vocab size   : {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
