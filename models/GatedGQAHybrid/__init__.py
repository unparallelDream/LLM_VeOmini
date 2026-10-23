# Independent GQA family (model_type gated_gqa_hybrid). Thin subclasses of the
# GatedMLAHybrid family: GQAAttention lives in the sibling package to avoid a
# circular import; everything else (hybrid/iHC/mHC/MoE/cache protocol) is
# shared through inheritance.

from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


@MODELING_REGISTRY.register("gated_gqa_hybrid")
def register_gated_gqa_hybrid_modeling(architecture: str):
    from .modeling_gated_gqa_hybrid import GatedGQAHybridForCausalLM, GatedGQAHybridModel

    if "ForCausalLM" in architecture:
        return GatedGQAHybridForCausalLM
    if "Model" in architecture:
        return GatedGQAHybridModel
    return GatedGQAHybridForCausalLM


@MODEL_CONFIG_REGISTRY.register("gated_gqa_hybrid")
def register_gated_gqa_hybrid_config():
    from .configuration_gated_gqa_hybrid import GatedGQAHybridConfig

    return GatedGQAHybridConfig
