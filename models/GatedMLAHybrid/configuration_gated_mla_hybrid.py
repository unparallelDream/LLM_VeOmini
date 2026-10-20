from __future__ import annotations

from transformers import PretrainedConfig


class GatedMLAHybridConfig(PretrainedConfig):
    """Configuration shared by the four token-mixer baselines."""

    model_type = "gated_mla_hybrid"
    architectures = ["GatedMLAHybridForCausalLM"]
    keys_to_ignore_at_inference = ["past_key_values"]

    BASELINES = {
        "gated_mla": (),
        "hils_gated_mla": (3, 7, 11, 15),
        "kda_gated_mla": (3, 7, 11, 15),
        "swa_gated_mla": (3, 7, 11, 15),
    }

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 384,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 6,
        head_dim: int = 64,
        intermediate_size: int = 1536,
        baseline: str = "gated_mla",
        linear_layer_indices: list[int] | tuple[int, ...] | None = None,
        q_lora_rank: int = 128,
        kv_lora_rank: int = 128,
        qk_nope_head_dim: int = 32,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        use_nope_mla: bool = False,
        attention_sink: bool = True,
        attention_sink_size: int = 1,
        window_size: int = 128,
        output_gate_type: str = "vector",
        output_gate_init: float = -2.0,
        use_mhc: bool = False,
        use_ihc: bool = False,
        ihc_num_streams: int = 4,
        ihc_alpha_init: float = 0.01,
        mhc_num_streams: int = 4,
        mhc_alpha_init: float = 0.01,
        mhc_sinkhorn_iters: int = 20,
        mhc_res_bias_init: float = 0.0,
        use_engram: bool = False,
        engram_vocab_size: int = 0,
        engram_dim: int = 0,
        channel_mixer: str = "swiglu",
        num_experts: int = 0,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: int = 0,
        shared_expert_intermediate_size: int = 0,
        moe_balance_loss_weight: float = 0.01,
        use_cache: bool = True,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        fuse_norm: bool = False,
        fuse_cross_entropy: bool = False,
        fuse_linear_cross_entropy: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if baseline not in self.BASELINES:
            raise ValueError(f"Unknown baseline {baseline!r}; choose from {sorted(self.BASELINES)}")
        if linear_layer_indices is None:
            linear_layer_indices = self.BASELINES[baseline]
        linear_layer_indices = list(linear_layer_indices)
        if any(i < 0 or i >= num_hidden_layers for i in linear_layer_indices):
            raise ValueError("linear_layer_indices contains an invalid layer index")
        if num_attention_heads * head_dim != hidden_size:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if q_lora_rank < 0 or kv_lora_rank <= 0:
            raise ValueError("q_lora_rank must be non-negative and kv_lora_rank must be positive")
        if output_gate_type not in {"vector"}:
            raise ValueError("Only vector output gates are supported")
        if channel_mixer not in {"swiglu"}:
            raise ValueError("Only swiglu channel mixer is currently supported")
        if attention_sink_size < 0:
            raise ValueError("attention_sink_size must be non-negative")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if use_mhc and use_ihc:
            raise ValueError("use_mhc and use_ihc are mutually exclusive")
        if use_engram and (engram_vocab_size <= 0 or engram_dim <= 0):
            raise ValueError("engram_vocab_size and engram_dim are required when use_engram=True")
        if num_experts > 1 and not 1 <= num_experts_per_tok <= num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts] when MoE is enabled")
        if num_experts == 1:
            raise ValueError("num_experts=1 is pointless; use dense (num_experts=0) instead")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.baseline = baseline
        self.linear_layer_indices = linear_layer_indices
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.use_nope_mla = use_nope_mla
        self.attention_sink = attention_sink
        self.attention_sink_size = attention_sink_size
        self.window_size = window_size
        self.output_gate_type = output_gate_type
        self.output_gate_init = output_gate_init
        self.use_mhc = use_mhc
        self.use_ihc = use_ihc
        # iHC (Identity Hyper-Connections): X_{l+1} = X + C*F(A*X) over
        # ihc_num_streams parallel residual streams; B is fixed to identity.
        # Coefficient generators follow DeepSeek-V4's dynamic parameterization
        # with small-initialized gating factors so training starts from the
        # standard-residual behavior.
        self.ihc_num_streams = ihc_num_streams
        self.ihc_alpha_init = ihc_alpha_init
        # mHC (Manifold-Constrained Hyper-Connections): X_{l+1} = B·X + C·F(A·X)
        # with B projected onto the Birkhoff polytope via Sinkhorn-Knopp on
        # exp(B_tilde). S_res = -tau*(J-I): tau=0 starts B at the uniform matrix
        # J/n (exact residual equivalence while streams are identical, healthy
        # gradients); tau>0 pulls B toward the identity at the cost of an
        # exp(-tau) gradient suppression on the off-diagonal.
        self.mhc_num_streams = mhc_num_streams
        self.mhc_alpha_init = mhc_alpha_init
        self.mhc_sinkhorn_iters = mhc_sinkhorn_iters
        self.mhc_res_bias_init = mhc_res_bias_init
        self.use_engram = use_engram
        # MoE (tier-1 eager reference): num_experts=0 keeps the dense SwiGLU;
        # expert width defaults to intermediate_size // top_k so active params
        # match the dense counterpart.
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size or (
            intermediate_size // max(1, num_experts_per_tok) if num_experts > 1 else 0
        )
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.moe_balance_loss_weight = moe_balance_loss_weight
        self.engram_vocab_size = engram_vocab_size
        self.engram_dim = engram_dim
        self.channel_mixer = channel_mixer
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
