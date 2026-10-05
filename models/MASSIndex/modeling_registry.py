"""Opt-in VeOmni registration for the standalone MASS model type."""

from veomni.models.loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODELING_REGISTRY.register("qwen_mass")
def register_qwen_mass_modeling(architecture: str):
    from .modeling_qwen_mass import MASSForCausalLM, MASSModel

    return MASSForCausalLM if "ForCausalLM" in architecture else MASSModel


@MODEL_CONFIG_REGISTRY.register("qwen_mass")
def register_qwen_mass_config():
    from .configuration_mass import MASSConfig

    return MASSConfig


@MODELING_REGISTRY.register("olmo_mass")
def register_olmo_mass_modeling(architecture: str):
    from .modeling_olmo_mass import MASSForCausalLM, MASSModel

    return MASSForCausalLM if "ForCausalLM" in architecture else MASSModel


@MODEL_CONFIG_REGISTRY.register("olmo_mass")
def register_olmo_mass_config():
    from .configuration_mass import OLMoMASSConfig

    return OLMoMASSConfig
