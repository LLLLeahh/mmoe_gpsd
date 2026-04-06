"""KuaiRec models (GPSD-style registration for gin)."""

from .args import ModelArgs
from .transformer import Transformer
from .mmoe_baseline import MMoEBaselineModel

__all__ = ["ModelArgs", "Transformer", "MMoEBaselineModel"]
