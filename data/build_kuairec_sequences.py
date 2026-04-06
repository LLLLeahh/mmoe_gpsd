"""
Build `pretrain_sequences.npz` for GPSD-style Transformer pretraining (KuaiRec).

Uses positive clicks only from processed `train.csv`, groups by user, sorts by time,
right-pads/truncates to `max_seq_len`. Pad token for items = max(video_id_idx)+1.

Usage:
  python data/build_kuairec_sequences.py --train_csv data/processed/train.csv --out data/processed/pretrain_sequences.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def build_sequences(
    train_csv: str,
    out_path: str,
    max_seq_len: int = 200,
) -> None:
    df = pd.read_csv(train_csv)
    if "click" in df.columns:
        df = df[df["click"] == 1]
    df = df.sort_values(["user_id_idx", "timestamp"])

    item_rows: list[np.ndarray] = []
    cate_rows: list[np.ndarray] = []

    max_vid = int(df["video_id_idx"].max()) if len(df) else 0
    pad_item = max_vid + 1
    item_vocab_size = max_vid + 2

    if "tag1" in df.columns:
        max_tag = int(df["tag1"].max())
        pad_cate = max_tag + 1
        cate_vocab_size = max_tag + 2
    else:
        pad_cate = 1
        cate_vocab_size = 2

    for _, g in df.groupby("user_id_idx"):
        seq = g["video_id_idx"].astype(np.int32).values
        if len(seq) < 2:
            continue
        if "tag1" in g.columns:
            cseq = g["tag1"].astype(np.int32).values
        else:
            cseq = np.zeros_like(seq)

        if len(seq) > max_seq_len:
            seq = seq[-max_seq_len:]
            cseq = cseq[-max_seq_len:]

        pad_len = max_seq_len - len(seq)
        if pad_len > 0:
            seq = np.concatenate([seq, np.full(pad_len, pad_item, dtype=np.int32)])
            cseq = np.concatenate([cseq, np.full(pad_len, pad_cate, dtype=np.int32)])

        item_rows.append(seq)
        cate_rows.append(cseq)

    if not item_rows:
        raise ValueError("No sequences with length >= 2; check train.csv")

    historical_item_ids = np.stack(item_rows, axis=0)
    historical_cate_ids = np.stack(cate_rows, axis=0)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        out_path,
        historical_item_ids=historical_item_ids,
        historical_cate_ids=historical_cate_ids,
        item_vocab_size=np.int32(item_vocab_size),
        cate_vocab_size=np.int32(cate_vocab_size),
    )
    print(
        f"Saved {out_path} shape item={historical_item_ids.shape} "
        f"item_vocab_size={item_vocab_size} cate_vocab_size={cate_vocab_size}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, default="data/processed/train.csv")
    p.add_argument("--out", type=str, default="data/processed/pretrain_sequences.npz")
    p.add_argument("--max_seq_len", type=int, default=200)
    args = p.parse_args()
    build_sequences(args.train_csv, args.out, max_seq_len=args.max_seq_len)


if __name__ == "__main__":
    main()
