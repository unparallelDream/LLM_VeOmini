from ..GatedMLAHybrid.configuration_gated_mla_hybrid import GatedMLAHybridConfig


class GatedGQAHybridConfig(GatedMLAHybridConfig):
    """GQA variant family (model_type ``gated_gqa_hybrid``).

    Thin subclass of GatedMLAHybridConfig pinning baseline='gated_gqa': the
    KV organization is grouped-query (num_key_value_heads groups,
    Qwen3.8-FlashNext / MiniMax-M3 style: few KV heads + per-head QK
    RMSNorm + partial RoPE) instead of GatedMLA's shared low-rank latent.
    Hybrid layers, iHC/mHC, MoE and the scaling fields are inherited
    unchanged, so MLA-vs-GQA stays a controlled single-variable comparison.
    """

    model_type = "gated_gqa_hybrid"
    architectures = ["GatedGQAHybridForCausalLM"]

    def __init__(self, baseline: str = "gated_gqa", **kwargs):
        if baseline != "gated_gqa":
            raise ValueError("GatedGQAHybridConfig fixes baseline='gated_gqa'")
        super().__init__(baseline="gated_gqa", **kwargs)
