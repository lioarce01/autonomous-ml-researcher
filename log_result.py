"""
log_result.py — CLI for Claude Code to log experiment results.

Usage:
    python log_result.py --name "baseline" --val_bpb 1.5 --notes "initial nanoGPT"
"""

import argparse
import subprocess
import sys
import db
import context_gen


def get_git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Log an experiment result to the DB.")
    parser.add_argument("--name", required=True, help="Experiment name (short, descriptive)")
    parser.add_argument("--val_bpb", required=True, type=float, help="Validation bits per byte")
    parser.add_argument("--notes", default="", help="Short description of what changed")
    parser.add_argument("--hypothesis", default=None,
        help="What you predicted would happen and why (optional)")
    args = parser.parse_args()

    row = db.log(args.name, args.val_bpb, args.notes, hypothesis=args.hypothesis)

    git_hash = get_git_hash()
    if git_hash:
        db.update_git_hash(row["id"], git_hash)
        row["git_hash"] = git_hash

    context_gen.generate()

    kept_str = "YES" if row["kept"] else "NO"
    stats = db.get_stats()
    best = stats["best_val_bpb"]

    print(f"\n[OK] Logged experiment #{row['id']}: {args.name}")
    print(f"  Val bpb  : {args.val_bpb:.6f}")
    print(f"  Kept     : {kept_str}")
    print(f"  Best so far: {best:.6f}")
    if git_hash:
        print(f"  Git hash : {git_hash}")
    print(f"\nCONTEXT.md updated.\n")

    # Suggested commit message
    action = "improve" if row["kept"] else "try"
    print(f"Suggested commit message:")
    print(f'  git commit -am "exp: {action} {args.name} -- val_bpb={args.val_bpb:.4f}"')
    print()


if __name__ == "__main__":
    main()
