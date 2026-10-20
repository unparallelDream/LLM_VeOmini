# Ported from LLM_Standardization/custom_models/gated_mla_hybrid (flame stack)
# to the VeOmni registry. Model code is unchanged; only the registration layer
# differs: AutoConfig.register -> MODEL_CONFIG_REGISTRY / MODELING_REGISTRY.
# KDA layers require flash-linear-attention (pip install --no-deps fla-core==0.5.2
# flash-linear-attention==0.5.2) and FLA_TILELANG=0 to force the triton backend
# (this env's tilelang pin is not the one fla's tilelang backend expects).

from veomni.models.loader import MODELING_REGISTRY, MODEL_CONFIG_REGISTRY


@MODELING_REGISTRY.register("gated_mla_hybrid")
def register_gated_mla_hybrid_modeling(architecture: str):
    from .modeling_gated_mla_hybrid import GatedMLAHybridForCausalLM, GatedMLAHybridModel

    if "ForCausalLM" in architecture:
        return GatedMLAHybridForCausalLM
    if "Model" in architecture:
        return GatedMLAHybridModel
    return GatedMLAHybridForCausalLM


@MODEL_CONFIG_REGISTRY.register("gated_mla_hybrid")
def register_gated_mla_hybrid_config():
    from .configuration_gated_mla_hybrid import GatedMLAHybridConfig

    return GatedMLAHybridConfig
