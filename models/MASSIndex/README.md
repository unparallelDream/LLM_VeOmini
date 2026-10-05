# MASS-Index MVP

This directory is intentionally independent from `FlashHiLS`, `ops/`, and the
existing JSON configurations.  It implements the first research milestone:

1. take a block-conditioned base anchor `a[c,0]`;
2. construct `M` low-rank-offset anchors `a[c,m]`;
3. pool one Taylor tangent `(mu[c,m], b[c,m])` per anchor;
4. score chunks with the upper envelope `max_m(q·mu[c,m]/sqrt(d) + b[c,m])`;
5. emit GQA-compatible top-k indices through a PyTorch reference path.

The current scope deliberately excludes fused kernels, cache serialization,
MoE-style balancing losses, and model registration.  `mass_topk_reference`
keeps the candidate-index shape `[B, L, Hkv, K]`, so a later independent
`MASSAttention` wrapper can call the existing sparse V-attention operator
without modifying it.

`mass_attention.py` now provides that separate `MASSAttention` wrapper for
training/prefill.  Instantiate it with the existing HiLS-compatible model
configuration plus a programmatic `MASSIndexConfig`; no existing config file
or implementation is changed.  Cache continuation is deliberately not part
of this first version.

For training, use the new `qwen_mass` wrapper via
`tasks/pretrain_mass_with_ruler.py` and
`configs/mass_attention/config_mass_attn_8KA2K_HoPE_345M_m4_r8.json`.
They are new files; the original HiLS launcher and JSON configurations are
not edited.
