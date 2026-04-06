"""
GPSD-style generative Transformer (TensorFlow/Keras).

Pretraining: causal LM on item (and optional category) sequences.
Return dict keys align with gpsd-rec-main/src/model/transformer.py: loss, item_ar_loss, cate_ar_loss, rank_loss.
"""

from __future__ import annotations

import gin
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .args import ModelArgs


def _causal_valid_mask(seq: tf.Tensor, pad_id: int = 0) -> tf.Tensor:
    """[B, L] float32 1.0 where valid (non-pad) token."""
    return tf.cast(tf.not_equal(seq, pad_id), tf.float32)


@gin.configurable
class Transformer(keras.Model):
    """Causal Transformer for next-item / next-cate prediction (AR pretraining)."""

    def __init__(self, params: ModelArgs, seed: int = 0):
        super().__init__()
        self.params = params
        self.seed = seed
        emb_dim = params.embedding_dim or params.dim
        d_model = params.dim
        assert params.item_vocab_size is not None
        cate_vocab_size = params.cate_vocab_size or 2
        self.item_embeddings = layers.Embedding(
            params.item_vocab_size,
            emb_dim,
            name="item_embeddings",
        )
        self.cate_embeddings = layers.Embedding(
            cate_vocab_size,
            emb_dim,
            name="cate_embeddings",
        )
        self.segment_embeddings = layers.Embedding(
            params.segment_vocab_size,
            emb_dim,
            name="segment_embeddings",
        )
        if emb_dim != d_model:
            self.pre_projector = layers.Dense(d_model, use_bias=False, name="pre_projector")
            self.post_projector = layers.Dense(emb_dim, use_bias=False, name="post_projector")
        else:
            self.pre_projector = None
            self.post_projector = None

        self.blocks = []
        dff = max(4 * d_model, (4 * d_model // params.multiple_of) * params.multiple_of)
        for i in range(params.n_layers):
            self.blocks.append(
                _TransformerBlock(
                    d_model,
                    params.n_heads,
                    dff,
                    params.dropout,
                    params.norm_eps,
                    name=f"block_{i}",
                )
            )
        self.norm = layers.LayerNormalization(epsilon=params.norm_eps, name="final_norm")
        self.dropout = layers.Dropout(params.dropout)
        self.item_logits_head = layers.Dense(params.item_vocab_size, name="item_logits_head")
        self.cate_logits_head = layers.Dense(params.cate_vocab_size, name="cate_logits_head")
        self.rank_dense = layers.Dense(1, name="rank_dense")

        tf.random.set_seed(seed)

    def call(self, inputs, training: bool = False, mask=None):
        """
        inputs dict:
          historical_item_ids [B, L]
          historical_cate_ids [B, L] optional
          historical_len [B] optional for rank head
          click_label [B] optional for rank
          item_ar_labels [B, L] optional (else next = roll(item_ids))
        """
        item_ids = inputs["historical_item_ids"]
        cate_ids = inputs.get("historical_cate_ids")
        if cate_ids is None:
            cate_ids = tf.zeros_like(item_ids)
        seg_ids = tf.zeros_like(item_ids)

        x = (
            self.item_embeddings(item_ids)
            + self.cate_embeddings(cate_ids)
            + self.segment_embeddings(seg_ids)
        )
        if self.pre_projector is not None:
            x = self.pre_projector(x)
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        h = x
        if self.post_projector is not None:
            x = self.post_projector(x)

        item_logits = self.item_logits_head(h)
        cate_logits = self.cate_logits_head(h)

        pad_m = _causal_valid_mask(item_ids)
        item_labels = inputs.get("item_ar_labels")
        if item_labels is None:
            item_labels = tf.roll(item_ids, shift=-1, axis=1)

        item_ar_loss = tf.constant(0.0, dtype=tf.float32)
        cate_ar_loss = tf.constant(0.0, dtype=tf.float32)
        rank_loss = tf.constant(0.0, dtype=tf.float32)
        rank_outputs = tf.constant(0.0, dtype=tf.float32)

        if self.params.item_ar_loss_weight > 0:
            logits = item_logits[:, :-1, :]
            lab = item_labels[:, :-1]
            valid = pad_m[:, 1:] * pad_m[:, :-1]
            lab = tf.where(lab < 0, tf.zeros_like(lab), lab)
            lab = tf.cast(lab, tf.int32)
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab, logits=logits)
            item_ar_loss = _masked_mean(ce, valid)

        if self.params.cate_ar_loss_weight > 0:
            cate_next = tf.roll(cate_ids, shift=-1, axis=1)
            logits_c = cate_logits[:, :-1, :]
            lab_c = cate_next[:, :-1]
            valid_c = pad_m[:, 1:] * pad_m[:, :-1]
            lab_c = tf.cast(lab_c, tf.int32)
            ce_c = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lab_c, logits=logits_c)
            cate_ar_loss = _masked_mean(ce_c, valid_c)

        if self.params.rank_loss_weight > 0 and "click_label" in inputs:
            hlen = inputs.get("historical_len")
            if hlen is not None:
                idx = tf.minimum(tf.cast(hlen, tf.int32), tf.shape(h)[1] - 1)
                batch = tf.range(tf.shape(h)[0])
                pooled = tf.gather_nd(h, tf.stack([batch, idx], axis=1))
                logits_r = tf.squeeze(self.rank_dense(pooled), -1)
                y = tf.cast(inputs["click_label"], tf.float32)
                rank_loss = tf.reduce_mean(
                    keras.losses.binary_crossentropy(y, logits_r, from_logits=True)
                )
                rank_outputs = tf.nn.sigmoid(logits_r)

        loss = (
            self.params.item_ar_loss_weight * item_ar_loss
            + self.params.cate_ar_loss_weight * cate_ar_loss
            + self.params.rank_loss_weight * rank_loss
        )
        return {
            "loss": loss,
            "item_ar_loss": item_ar_loss,
            "cate_ar_loss": cate_ar_loss,
            "rank_loss": rank_loss,
            "rank_outputs": rank_outputs,
        }

    def compile(self, optimizer="adam", **kwargs):
        super().compile(optimizer=optimizer, loss=None, metrics=None)

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            out = self(x, training=True)
            loss = out["loss"]
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {
            "loss": loss,
            "item_ar_loss": out["item_ar_loss"],
            "cate_ar_loss": out["cate_ar_loss"],
            "rank_loss": out["rank_loss"],
        }

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        out = self(x, training=False)
        return {
            "loss": out["loss"],
            "item_ar_loss": out["item_ar_loss"],
            "cate_ar_loss": out["cate_ar_loss"],
            "rank_loss": out["rank_loss"],
        }


def _masked_mean(ce: tf.Tensor, valid: tf.Tensor) -> tf.Tensor:
    w = tf.cast(valid, tf.float32)
    return tf.reduce_sum(ce * w) / (tf.reduce_sum(w) + 1e-8)


class _TransformerBlock(layers.Layer):
    def __init__(self, d_model: int, n_heads: int, dff: int, dropout: float, eps: float, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout,
        )
        self.ffn = keras.Sequential(
            [
                layers.Dense(dff, activation="relu"),
                layers.Dense(d_model),
            ]
        )
        self.ln1 = layers.LayerNormalization(epsilon=eps)
        self.ln2 = layers.LayerNormalization(epsilon=eps)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True,
        )
        x = self.ln1(x + self.drop1(attn, training=training))
        f = self.ffn(x)
        x = self.ln2(x + self.drop2(f, training=training))
        return x
