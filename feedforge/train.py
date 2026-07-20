"""Train BERT4Rec on MovieLens-1M.

Local (slow, fine for smoke tests):
    python -m feedforge.train --data ml-1m/ratings.dat --epochs 2

Colab GPU (real training run, ~30-60 min for 100+ epochs):
    python -m feedforge.train --data ml-1m/ratings.dat --epochs 200 \
        --d-model 64 --n-layers 2 --batch-size 256

Validation uses FULL-RANKING metrics on the held-out second-to-last item
per user (see evaluate.py for why not sampled metrics). Best checkpoint
by validation recall@10 is saved; final test metrics come from
scripts/report_metrics.py against the untouched last item.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import MLMSequenceDataset, build_sequences, load_movielens_1m
from .evaluate import full_ranking_metrics, popularity_baseline
from .model import BERT4Rec


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(args: argparse.Namespace) -> None:
    device = args.device or pick_device()
    print(f"device: {device}")

    df = load_movielens_1m(args.data)
    split = build_sequences(df)
    print(f"users: {len(split.train):,}  items: {split.n_items:,}")

    dataset = MLMSequenceDataset(
        split.train,
        n_items=split.n_items,
        max_len=args.max_len,
        mask_prob=args.mask_prob,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )

    model = BERT4Rec(
        vocab_size=split.vocab_size,
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Baseline first: the number the model must beat to mean anything
    pop = popularity_baseline(split.train, split.train, split.valid_target, split.n_items)
    print(f"popularity baseline (valid): {pop}")

    best_recall = -1.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        for tokens, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = model(tokens)
            # Loss only on masked positions (labels==PAD elsewhere)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        msg = f"epoch {epoch:3d}  loss {epoch_loss / max(n_batches,1):.4f}  {time.time()-t0:.1f}s"

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = full_ranking_metrics(
                model,
                histories=split.train,
                targets=split.valid_target,
                n_items=split.n_items,
                mask_token=split.mask_token,
                max_len=args.max_len,
                device=device,
            )
            msg += f"  valid {metrics}"
            if metrics["recall@10"] > best_recall:
                best_recall = metrics["recall@10"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": vars(args),
                        "n_items": split.n_items,
                        "vocab_size": split.vocab_size,
                        "valid_metrics": metrics,
                    },
                    out_dir / "bert4rec_best.pt",
                )
                msg += "  [saved]"
        print(msg, flush=True)

    print(f"best valid recall@10: {best_recall:.4f}")
    print(f"checkpoint: {out_dir / 'bert4rec_best.pt'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to ml-1m/ratings.dat")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--mask-prob", type=float, default=0.2)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto if omitted)")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
