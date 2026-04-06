"""Model hyperparameters (GPSD-style gin-configurable ModelArgs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gin


@gin.configurable
@dataclass
class ModelArgs:
    dim: int = 64
    embedding_dim: Optional[int] = None
    n_layers: int = 4
    n_heads: int = 4
    item_vocab_size: Optional[int] = None
    cate_vocab_size: Optional[int] = None
    segment_vocab_size: int = 2
    multiple_of: int = 32
    norm_eps: float = 1e-5
    max_seq_len: int = 200
    dropout: float = 0.0
    rank_loss_weight: float = 1.0
    use_causal_mask: bool = True
    n_samples: int = 4096
    temperature: float = 0.05
    l2_norm: bool = True
    item_ar_loss_weight: float = 0.0
    cate_ar_loss_weight: float = 0.0
    attention_type: str = "bilinear_attention"
    pos_emb_dim: int = 32
    # KuaiRec MMoE discriminative (flat features)
    num_dense_features: int = 32
    mmoe_units: int = 64
    num_experts: int = 8
    tower_units: int = 64
    num_tasks: int = 2
    use_pretrained_item_embedding: bool = False
    pretrained_embedding_dim: Optional[int] = None
