"""Independent configuration type for the MASS Qwen wrapper."""

from models.FlashHiLS.configuration_hils import HiLSConfig


class MASSConfig(HiLSConfig):
    """HiLS-compatible configuration with MASS-only hyperparameters.

    This class is separate from ``HiLSConfig`` so existing configuration files
    and model registrations retain their exact behaviour.
    """

    model_type = "qwen_mass"

    def __init__(
        self,
        mass_num_anchors: int = 4,
        mass_offset_rank: int = 8,
        mass_anchor_rms_norm: bool = True,
        mass_offset_init_std: float = 0.02,
        **kwargs,
    ):
        self.mass_num_anchors = mass_num_anchors
        self.mass_offset_rank = mass_offset_rank
        self.mass_anchor_rms_norm = mass_anchor_rms_norm
        self.mass_offset_init_std = mass_offset_init_std
        super().__init__(**kwargs)


class OLMoMASSConfig(MASSConfig):
    """OLMo3/HiLS-compatible MASS configuration."""

    model_type = "olmo_mass"
