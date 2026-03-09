"""
prepare.py — READ ONLY. Downloads and tokenizes TinyShakespeare once.

Creates data/input.bin (uint16 token array).
Run once before training: python prepare.py
"""

import os
import struct
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_BIN = os.path.join(DATA_DIR, "input.bin")
INPUT_TXT = os.path.join(DATA_DIR, "input.txt")

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(INPUT_TXT):
        print(f"input.txt already exists ({os.path.getsize(INPUT_TXT):,} bytes), skipping download.")
        return
    print(f"Downloading TinyShakespeare from {SHAKESPEARE_URL} ...")
    urllib.request.urlretrieve(SHAKESPEARE_URL, INPUT_TXT)
    print(f"Downloaded {os.path.getsize(INPUT_TXT):,} bytes → {INPUT_TXT}")


def tokenize():
    if os.path.exists(INPUT_BIN):
        print(f"input.bin already exists ({os.path.getsize(INPUT_BIN):,} bytes), skipping tokenize.")
        return

    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        text = f.read()

    # Character-level tokenization (same as karpathy/nanoGPT char model)
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}

    tokens = [stoi[c] for c in text]
    n = len(tokens)

    # Write header: [vocab_size (4 bytes), n_tokens (4 bytes)] then uint16 tokens
    with open(INPUT_BIN, "wb") as f:
        f.write(struct.pack("<II", vocab_size, n))
        for t in tokens:
            f.write(struct.pack("<H", t))

    print(f"Tokenized {n:,} characters → vocab size {vocab_size}")
    print(f"Wrote {os.path.getsize(INPUT_BIN):,} bytes → {INPUT_BIN}")

    # Save vocab for reference
    vocab_path = os.path.join(DATA_DIR, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chars):
            f.write(f"{i}\t{repr(c)}\n")
    print(f"Vocab saved → {vocab_path}")


if __name__ == "__main__":
    download()
    tokenize()
    print("\nDone. You can now run: python train.py")
