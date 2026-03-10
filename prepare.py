"""
prepare.py — Downloads TinyStories, trains BPE tokenizer (vocab=8192), tokenizes to binary.
Run once before training. Outputs:
  data/input.bin      — uint16 tokens (train split)
  data/val.bin        — uint16 tokens (val split)
  data/meta.json      — vocab_size, avg_bytes_per_token (needed for val_bpb)
  data/tokenizer.json — saved BPE tokenizer (for inspection/reuse)
"""
import os, json, struct
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VOCAB_SIZE = 8192
DATASET_FRACTION = "10%"   # ~200MB text -- enough data, fast download


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Download TinyStories (train + validation splits)
    print("Downloading TinyStories...")
    train_ds = load_dataset("roneneldan/TinyStories", split=f"train[:{DATASET_FRACTION}]")
    val_ds   = load_dataset("roneneldan/TinyStories", split="validation[:5%]")
    train_texts = train_ds["text"]
    val_texts   = val_ds["text"]
    print(f"  Train stories: {len(train_texts):,} | Val stories: {len(val_texts):,}")

    # 2. Train BPE tokenizer on train text
    print(f"Training BPE tokenizer (vocab={VOCAB_SIZE})...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]"])
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    tokenizer.save(os.path.join(DATA_DIR, "tokenizer.json"))
    print(f"  Tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")

    # 3. Tokenize train and compute avg_bytes_per_token
    print("Tokenizing train split...")
    all_tokens = []
    total_bytes = 0
    for text in train_texts:
        enc = tokenizer.encode(text)
        all_tokens.extend(enc.ids)
        total_bytes += len(text.encode("utf-8"))
    avg_bytes_per_token = total_bytes / len(all_tokens)

    # Write train binary: header = (vocab_size, n_tokens) as uint32, then uint16 tokens
    input_bin = os.path.join(DATA_DIR, "input.bin")
    with open(input_bin, "wb") as f:
        f.write(struct.pack("<II", tokenizer.get_vocab_size(), len(all_tokens)))
        for t in all_tokens:
            f.write(struct.pack("<H", t))

    # 4. Tokenize val split
    print("Tokenizing val split...")
    val_tokens = []
    for text in val_texts:
        val_tokens.extend(tokenizer.encode(text).ids)
    val_bin = os.path.join(DATA_DIR, "val.bin")
    with open(val_bin, "wb") as f:
        f.write(struct.pack("<II", tokenizer.get_vocab_size(), len(val_tokens)))
        for t in val_tokens:
            f.write(struct.pack("<H", t))

    # 5. Write meta.json
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "avg_bytes_per_token": avg_bytes_per_token,
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone.")
    print(f"  Train tokens : {len(all_tokens):,}")
    print(f"  Val tokens   : {len(val_tokens):,}")
    print(f"  avg bytes/tok: {avg_bytes_per_token:.3f}")
    print(f"  Vocab size   : {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
