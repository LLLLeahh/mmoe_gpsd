"""
GPSD-style training entry (gin + train.model_cls), TensorFlow implementation.

Usage (from project root):
  set PYTHONPATH=src
  python src/train.py --config config/kuairec/transformer_pretrain.gin
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional, Type

import gin
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from kuairec_data import load_feature_config, make_pretrain_tf_dataset
from model import MMoEBaselineModel, ModelArgs, Transformer
from model.mmoe_baseline import compile_baseline


def _latest_weights_h5(ckpt_dir: str) -> Optional[str]:
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None
    best = []
    for name in os.listdir(ckpt_dir):
        if name.endswith(".weights.h5") or name.endswith(".h5"):
            best.append(os.path.join(ckpt_dir, name))
    if not best:
        return None
    return max(best, key=os.path.getmtime)


def _apply_frozen_regex(model: tf.keras.Model, frozen_params: Optional[str]) -> None:
    if not frozen_params:
        return
    pat = re.compile(frozen_params)

    def walk(m):
        for layer in m.layers:
            if pat.search(layer.name):
                layer.trainable = False
            if getattr(layer, "layers", None):
                walk(layer)

    walk(model)


def _load_weights_filtered(
    model: tf.keras.Model,
    ckpt_path: str,
    load_params: str,
) -> None:
    if not ckpt_path:
        return
    path = ckpt_path
    if os.path.isdir(path):
        path = _latest_weights_h5(path) or path
    if not path or not os.path.exists(path):
        print(f"[train] load_ckpt missing or empty: {ckpt_path}")
        return
    pat = re.compile(load_params)
    try:
        model.load_weights(path, by_name=True, skip_mismatch=True)
    except Exception as e:
        print(f"[train] full load_weights failed ({e}), trying layer-wise...")
        for layer in model.layers:
            if pat.search(layer.name):
                try:
                    w_path = path
                    layer.load_weights(w_path)
                except Exception:
                    pass


@gin.configurable
def train(
    epoch: int = 10,
    batch_size: int = 256,
    data_path: Optional[str] = None,
    eval_data_path: Optional[str] = None,
    ckpt_root_dir: str = "./output/kuairec/",
    ckpt_name: str = "run",
    learning_rate: float = 1e-3,
    min_learning_rate: Optional[float] = None,
    weight_decay: float = 0.0,
    shuffle: bool = True,
    load_ckpt: Optional[str] = None,
    load_params: str = ".*",
    frozen_params: Optional[str] = None,
    model_cls: Type = Transformer,
    random_seed: int = 42,
    processed_dir: str = "data/processed/",
    mmoe_use_feature_config: bool = True,
    mmoe_feature_cols: Optional[str] = None,
    early_stopping_patience: int = 0,
):
    tf.random.set_seed(random_seed)
    os.makedirs(os.path.join(ckpt_root_dir, ckpt_name), exist_ok=True)
    save_dir = os.path.join(ckpt_root_dir, ckpt_name)

    model_args = ModelArgs()

    if model_cls is Transformer:
        if not data_path:
            raise ValueError("Transformer pretrain requires train.data_path to .npz")
        ds_train, meta = make_pretrain_tf_dataset(
            data_path, batch_size=batch_size, shuffle=shuffle, seed=random_seed
        )
        ds_val = None
        if eval_data_path and os.path.isfile(eval_data_path):
            ds_val, _ = make_pretrain_tf_dataset(
                eval_data_path, batch_size=batch_size, shuffle=False, seed=random_seed
            )

        model_args.item_vocab_size = int(meta["item_vocab_size"])
        model_args.cate_vocab_size = int(meta["cate_vocab_size"])

        model = model_cls(model_args, seed=random_seed)
        opt = Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt)

        if load_ckpt:
            _load_weights_filtered(model, load_ckpt, load_params)
        _apply_frozen_regex(model, frozen_params)

        cb = [
            ModelCheckpoint(
                filepath=os.path.join(save_dir, "weights-{epoch:02d}.weights.h5"),
                save_weights_only=True,
                save_freq="epoch",
            )
        ]
        fit_kwargs = {
            "x": ds_train,
            "epochs": epoch,
            "callbacks": cb,
        }
        if ds_val is not None:
            fit_kwargs["validation_data"] = ds_val
        model.fit(**fit_kwargs)
        model.save_weights(os.path.join(save_dir, "weights_final.weights.h5"))
        print("[train] Transformer pretrain done, saved:", save_dir)
        return

    if model_cls is MMoEBaselineModel:
        import numpy as np
        import pandas as pd

        if mmoe_use_feature_config:
            sparse, dense = load_feature_config(processed_dir)
            feature_cols = sparse + dense
        else:
            if not mmoe_feature_cols:
                raise ValueError("Set train.mmoe_feature_cols or train.mmoe_use_feature_config=True")
            feature_cols = [c.strip() for c in mmoe_feature_cols.split(",") if c.strip()]

        train_csv = data_path or os.path.join(processed_dir, "train.csv")
        val_csv = eval_data_path or os.path.join(processed_dir, "val.csv")
        use_vid = model_args.item_vocab_size is not None
        if use_vid and "video_id_idx" in feature_cols:
            feature_cols = [c for c in feature_cols if c != "video_id_idx"]

        model_args.num_dense_features = len(feature_cols)

        model = model_cls(model_args, seed=random_seed)
        inner = model.inner

        if load_ckpt:
            _load_weights_filtered(inner, load_ckpt, load_params)
        _apply_frozen_regex(inner, frozen_params)

        compile_baseline(inner, learning_rate=learning_rate)

        cb = []
        if early_stopping_patience > 0 and os.path.isfile(val_csv):
            cb.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                )
            )
        cb.append(
            ModelCheckpoint(
                filepath=os.path.join(save_dir, "mmoe-{epoch:02d}.weights.h5"),
                save_weights_only=True,
                save_freq="epoch",
            )
        )

        df = pd.read_csv(train_csv)
        sw = {"value": df["click"].values.astype(np.float32)}
        y = {
            "ctr": df["click"].values.astype(np.float32),
            "value": df["watch_ratio"].values.astype(np.float32),
        }

        if use_vid:
            x = {
                "video_id_idx": df["video_id_idx"].values.astype(np.int32).reshape(-1, 1),
                "features": df[feature_cols].values.astype(np.float32),
            }
        else:
            x = df[feature_cols].values.astype(np.float32)

        val_data = None
        if os.path.isfile(val_csv):
            dv = pd.read_csv(val_csv)
            swv = {"value": dv["click"].values.astype(np.float32)}
            yv = {
                "ctr": dv["click"].values.astype(np.float32),
                "value": dv["watch_ratio"].values.astype(np.float32),
            }
            if use_vid:
                xv = {
                    "video_id_idx": dv["video_id_idx"].values.astype(np.int32).reshape(-1, 1),
                    "features": dv[feature_cols].values.astype(np.float32),
                }
            else:
                xv = dv[feature_cols].values.astype(np.float32)
            val_data = (xv, yv, swv)

        inner.fit(
            x,
            y,
            sample_weight=sw,
            validation_data=val_data,
            epochs=epoch,
            batch_size=batch_size,
            callbacks=cb,
            verbose=1,
        )
        inner.save_weights(os.path.join(save_dir, "mmoe_final.weights.h5"))
        print("[train] MMoE discriminative done, saved:", save_dir)
        return

    raise ValueError(f"Unsupported model_cls: {model_cls}")


def main():
    parser = argparse.ArgumentParser(description="KuaiRec GPSD-style training (gin)")
    parser.add_argument("--config", type=str, required=True, help="Path to .gin config file")
    args, unknown = parser.parse_known_args()
    gin.parse_config_file(args.config)
    if unknown:
        bindings = [u[2:] if u.startswith("--") else u for u in unknown]
        gin.parse_config(bindings)
    train()


if __name__ == "__main__":
    main()
