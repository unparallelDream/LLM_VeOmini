from ..GatedMLAHybrid.modeling_gated_mla_hybrid import (  # noqa: F401
    GatedMLAHybridForCausalLM,
    GatedMLAHybridModel,
    GQAAttention,
)
from .configuration_gated_gqa_hybrid import GatedGQAHybridConfig


class GatedGQAHybridModel(GatedMLAHybridModel):
    config_class = GatedGQAHybridConfig


class GatedGQAHybridForCausalLM(GatedMLAHybridForCausalLM):
    config_class = GatedGQAHybridConfig
