"""
MMoE multi-task baseline: CTR (binary) + continuous target (MAE loss).

运行:
    python mmoe_baseline.py

GPSD-style: `MMoEBaselineModel(model_args, seed)` for gin `train.model_cls=@MMoEBaselineModel`.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Tuple

import gin
import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Concatenate, Dense, Embedding, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

try:
    from .args import ModelArgs
    from .mmoe import MMoE
except ImportError:
    from args import ModelArgs
    from mmoe import MMoE

DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_mmoe_baseline(
    num_features: int,
    mmoe_units: int = 64,
    num_experts: int = 8,
    num_tasks: int = 2,
    tower_units: int = 64,
    expert_activation: str = "relu",
    gate_activation: str = "softmax",
) -> Model:
    """
    Shared MMoE -> per-task tower -> ctr (sigmoid) + value (linear).

    Task 0 name: ctr  — click, binary cross-entropy, AUC
    Task 1 name: value — regression, MAE loss
    """
    inp = Input(shape=(num_features,), name="features")

    mmoe_outputs = MMoE(
        units=mmoe_units,
        num_experts=num_experts,
        num_tasks=num_tasks,
        expert_activation=expert_activation,
        gate_activation=gate_activation,
    )(inp)

    task_names: List[str] = ["ctr", "value"]
    outputs = []
    for i, task_tensor in enumerate(mmoe_outputs):
        tower = Dense(
            tower_units,
            activation="relu",
            kernel_initializer=VarianceScaling(),
            name=f"tower_{task_names[i]}",
        )(task_tensor)
        if task_names[i] == "ctr":
            out = Dense(
                1,
                activation="sigmoid",
                kernel_initializer=VarianceScaling(),
                name="ctr",
            )(tower)
        else:
            out = Dense(
                1,
                activation="linear",
                kernel_initializer=VarianceScaling(),
                name="value",
            )(tower)
        outputs.append(out)

    return Model(inputs=inp, outputs=outputs, name="mmoe_ctr_value_baseline")


@gin.configurable
class MMoEBaselineModel:
    """
    Discriminative MMoE for gin `train.model_cls=@MMoEBaselineModel`.

    If `params.item_vocab_size` is set, builds `item_embeddings` + concat dense features
    (for GPSD-style transfer of pretrained item embeddings). Otherwise flat `features` only.
    Exposes `.inner` as the Keras `Model` used for compile/fit/evaluate.
    """

    def __init__(self, params: ModelArgs, seed: int = 0):
        self.params = params
        self.seed = seed
        set_seed(seed)
        if params.item_vocab_size is None:
            self.inner = build_mmoe_baseline(
                num_features=params.num_dense_features,
                mmoe_units=params.mmoe_units,
                num_experts=params.num_experts,
                num_tasks=params.num_tasks,
                tower_units=params.tower_units,
            )
        else:
            self.inner = _build_mmoe_with_item_embedding(params)


def _build_mmoe_with_item_embedding(params: ModelArgs) -> Model:
    emb_dim = params.pretrained_embedding_dim or params.dim
    vid_in = Input(shape=(1,), name="video_id_idx", dtype=tf.int32)
    feat_in = Input(shape=(params.num_dense_features,), name="features")
    emb = Embedding(
        params.item_vocab_size,
        emb_dim,
        name="item_embeddings",
    )(vid_in)
    emb = Flatten()(emb)
    inp = Concatenate(name="mmoe_input")([emb, feat_in])

    mmoe_outputs = MMoE(
        units=params.mmoe_units,
        num_experts=params.num_experts,
        num_tasks=params.num_tasks,
        expert_activation="relu",
        gate_activation="softmax",
    )(inp)

    task_names: List[str] = ["ctr", "value"]
    outputs = []
    for i, task_tensor in enumerate(mmoe_outputs):
        tower = Dense(
            params.tower_units,
            activation="relu",
            kernel_initializer=VarianceScaling(),
            name=f"tower_{task_names[i]}",
        )(task_tensor)
        if task_names[i] == "ctr":
            out = Dense(
                1,
                activation="sigmoid",
                kernel_initializer=VarianceScaling(),
                name="ctr",
            )(tower)
        else:
            out = Dense(
                1,
                activation="linear",
                kernel_initializer=VarianceScaling(),
                name="value",
            )(tower)
        outputs.append(out)

    return Model(
        inputs=[vid_in, feat_in],
        outputs=outputs,
        name="mmoe_ctr_value_with_item_emb",
    )


def synthetic_ctr_value_data(
    n_samples: int,
    n_features: int,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Synthetic: features ~ N(0,1); click from logit(linear + noise); value = f(features) + noise.

    仅用于打通训练流程，不代表真实分布。
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    w_click = rng.standard_normal(n_features)
    logits = x.dot(w_click) + 0.5 * x[:, 0] * x[:, 1]
    p = 1.0 / (1.0 + np.exp(-logits))
    y_ctr = (rng.random(n_samples) < p).astype(np.float32)
    w_val = rng.standard_normal(n_features)
    y_value = x.dot(w_val) + 0.3 * logits + rng.normal(0, 0.5, n_samples)
    y_value = y_value.astype(np.float32)

    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    sl_train = slice(0, n_train)
    sl_val = slice(n_train, n_train + n_val)
    sl_test = slice(n_train + n_val, None)

    y_train = {"ctr": y_ctr[sl_train], "value": y_value[sl_train]}
    y_val = {"ctr": y_ctr[sl_val], "value": y_value[sl_val]}
    y_test = {"ctr": y_ctr[sl_test], "value": y_value[sl_test]}

    return x, y_train, y_val, y_test


def compile_baseline(
    model: Model,
    learning_rate: float = 1e-3,
    ctr_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
) -> None:
    """
    Value 头使用 MAE；训练/验证/评估时需对 value 传入 sample_weight=click，
    使 MAE 与回归损失仅在正样本（点击）上累计。
    """
    try:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                "ctr": "binary_crossentropy",
                "value": "mean_absolute_error",
            },
            loss_weights={
                "ctr": ctr_loss_weight,
                "value": value_loss_weight,
            },
            metrics={
                "ctr": [
                    metrics.AUC(name="auc"),
                    metrics.BinaryAccuracy(name="accuracy"),
                ],
                "value": [],
            },
            weighted_metrics={
                "value": [metrics.MeanAbsoluteError(name="mae")],
            },
        )
    except TypeError:
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss={
                "ctr": "binary_crossentropy",
                "value": "mean_absolute_error",
            },
            loss_weights={
                "ctr": ctr_loss_weight,
                "value": value_loss_weight,
            },
            metrics={
                "ctr": [
                    metrics.AUC(name="auc"),
                    metrics.BinaryAccuracy(name="accuracy"),
                ],
                "value": [metrics.MeanAbsoluteError(name="mae")],
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MMoE baseline: CTR + value (MAE)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--mmoe-units", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--tower-units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ctr-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)

    x, y_train, y_val, y_test = synthetic_ctr_value_data(
        args.n_samples, args.n_features, seed=args.seed
    )
    n_train = y_train["ctr"].shape[0]
    n_val = y_val["ctr"].shape[0]
    x_train, x_val = x[:n_train], x[n_train : n_train + n_val]
    x_test = x[n_train + n_val :]

    model = build_mmoe_baseline(
        num_features=args.n_features,
        mmoe_units=args.mmoe_units,
        num_experts=args.num_experts,
        tower_units=args.tower_units,
    )
    compile_baseline(
        model,
        learning_rate=args.lr,
        ctr_loss_weight=args.ctr_weight,
        value_loss_weight=args.value_weight,
    )

    model.summary()

    es = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    # MAE / value 损失仅在 click==1 的样本上计算；ctr 头仍使用全量样本
    sw_train = {"value": y_train["ctr"].astype(np.float32)}
    sw_val = {"value": y_val["ctr"].astype(np.float32)}
    sw_test = {"value": y_test["ctr"].astype(np.float32)}

    model.fit(
        x_train,
        y_train,
        sample_weight=sw_train,
        validation_data=(x_val, y_val, sw_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[es],
        verbose=1,
    )

    results = model.evaluate(x_test, y_test, sample_weight=sw_test, verbose=0)
    print("Test metrics:", dict(zip(model.metrics_names, results)))


if __name__ == "__main__":
    main()
