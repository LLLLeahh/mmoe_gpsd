"""
KuaiRec datasets for GPSD-style training (outside gpsd-rec-main).

- Pretrain: .npz from `data/build_kuairec_sequences.py`
- Discriminative: processed train/val CSV from `data/data_preprocessing.py`
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


def load_pretrain_npz(path: str) -> Dict[str, np.ndarray | int]:
    d = np.load(path, allow_pickle=True)
    out = {
        "historical_item_ids": d["historical_item_ids"],
        "historical_cate_ids": d.get("historical_cate_ids"),
        "item_vocab_size": int(d["item_vocab_size"]),
        "cate_vocab_size": int(d["cate_vocab_size"]) if "cate_vocab_size" in d else 2,
    }
    if out["historical_cate_ids"] is None:
        out["historical_cate_ids"] = np.zeros_like(out["historical_item_ids"])
    return out


def make_pretrain_tf_dataset(
    npz_path: str,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, Dict[str, np.ndarray | int]]:
    data = load_pretrain_npz(npz_path)
    item = data["historical_item_ids"].astype(np.int32)
    cate = data["historical_cate_ids"].astype(np.int32)
    n = item.shape[0]
    ds = tf.data.Dataset.from_tensor_slices(
        {"historical_item_ids": item, "historical_cate_ids": cate}
    )
    if shuffle:
        ds = ds.shuffle(min(10000, n), seed=seed)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, data


def load_mmoe_csv(
    csv_path: str,
    feature_cols: List[str],
    label_click: str = "click",
    label_value: str = "watch_ratio",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    df = pd.read_csv(csv_path)
    X = df[feature_cols].values.astype(np.float32)
    y_ctr = df[label_click].values.astype(np.float32)
    y_val = df[label_value].values.astype(np.float32)
    vid = None
    if "video_id_idx" in df.columns and "video_id_idx" not in feature_cols:
        vid = df["video_id_idx"].values.astype(np.int32).reshape(-1, 1)
    return (
        {"features": X, "video_id_idx": vid} if vid is not None else {"features": X},
        {"ctr": y_ctr, "value": y_val},
    )


def load_feature_config(processed_dir: str) -> Tuple[List[str], List[str]]:
    path = os.path.join(processed_dir, "feature_config.pkl")
    with open(path, "rb") as f:
        fc = pickle.load(f)
    sparse = list(fc.get("sparse_features", []))
    dense = list(fc.get("dense_features", []))
    return sparse, dense


def make_mmoe_tf_dataset(
    train_csv: str,
    val_csv: Optional[str],
    feature_cols: List[str],
    batch_size: int,
    use_video_id: bool,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], int]:
    x_tr, y_tr = load_mmoe_csv(train_csv, feature_cols)
    n = x_tr["features"].shape[0]

    def pack_xy(xd, yd):
        if use_video_id and xd.get("video_id_idx") is not None:
            return (
                {"video_id_idx": xd["video_id_idx"], "features": xd["features"]},
                {"ctr": yd["ctr"], "value": yd["value"]},
            )
        return xd["features"], {"ctr": yd["ctr"], "value": yd["value"]}

    tx, ty = pack_xy(x_tr, y_tr)
    ds = tf.data.Dataset.from_tensor_slices((tx, ty))
    if shuffle:
        ds = ds.shuffle(min(10000, n), seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_val = None
    if val_csv and os.path.isfile(val_csv):
        x_v, y_v = load_mmoe_csv(val_csv, feature_cols)
        vx, vy = pack_xy(x_v, y_v)
        ds_val = tf.data.Dataset.from_tensor_slices((vx, vy)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds, ds_val, n
