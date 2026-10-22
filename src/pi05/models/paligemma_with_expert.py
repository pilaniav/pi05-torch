import torch
from torch import nn
from torch.utils.checkpoint import checkpoint 

from .gemma import GemmaModel, apply_rope, attention, rope_cos_sin
from .gemma_config import GemmaConfig
from .paligemma import PaliGemma
from .siglip_config import SiglipConfig

# Per-layer (k, v) of already processed tokens, keys already RoPE-rotated.
# Each tensor: (B, num_kv_heads, L_cached, head_dim)
KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class PaliGemmaWithExpert(nn.Module):
    """
    PaliGemma (VLM) + Gemma action expert with joint attention.

    forward        : training, prefix + suffix in one pass
    forward_prefix : inference, prefix only, returns a KV cache
    forward_suffix : inference, suffix only, attends to the cached prefix
    """

    def __init__(
        self, vision_config: SiglipConfig, vlm_config: GemmaConfig, expert_config: GemmaConfig
    ):
        super().__init__()
        for field in ("depth", "num_heads", "num_kv_heads", "head_dim", "rope_theta"):
            if getattr(vlm_config, field) != getattr(expert_config, field):
                raise ValueError(f"VLM and expert must share '{field}' for joint attention")

        self.paligemma = PaliGemma(vision_config, vlm_config)
        self.expert = GemmaModel(expert_config)
        self.gradient_checkpointing = False  # use set_gradient_checkpointing() to change

        # Note :
        #   Precision (bf16 weights, fp32 for selected params) is applied outside 
        #   this class, in the training/inference setup. Not done yet.

    def set_gradient_checkpointing(self, enabled: bool):
        """
        Training memory saver: recompute activations in backward instead of storing them.
        Covers the joint Gemma layers (our flag, used in _run) and SigLIP (HF's switch).
        """
        self.gradient_checkpointing = enabled
        vision = self.paligemma.vision.model                 # HF SiglipVisionModel
        if enabled:
            vision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            vision.gradient_checkpointing_disable()

    def forward(self, prefix_embs, suffix_embs, positions, mask, adarms_cond):
        """
        Training: joint pass over [prefix | suffix]

        prefix_embs (B, Lp, vlm width)
        suffix_embs (B, Ls, expert width)
        positions (B, Lp+Ls)
        mask (B, Lp+Ls, Lp+Ls) as bool type
        adarms_cond (B, expert width)
        returns prefix_out (B, Lp, vlm width), suffix_out (B, Ls, expert width)
        """
        streams = [(self.paligemma.llm, prefix_embs, None), (self.expert, suffix_embs, adarms_cond)]
        (prefix_out, suffix_out), _ = self._run(streams, positions, mask)
        return prefix_out, suffix_out

    def forward_prefix(self, prefix_embs, positions, mask):
        """
        Inference step 1: run the prefix through the VLM once and keep its K/V.
        positions (B, Lp), mask (B, Lp, Lp) -> prefix_out (B, Lp, vlm width), kv_cache
        """
        streams = [(self.paligemma.llm, prefix_embs, None)]
        (prefix_out,), kv_cache = self._run(streams, positions, mask, return_cache=True)
        return prefix_out, kv_cache

    def forward_suffix(self, suffix_embs, positions, mask, adarms_cond, kv_cache: KVCache):
        """
        Inference, each denoising step: expert on the suffix, attending to the cached prefix.
        positions (B, Ls) of the suffix tokens; mask (B, Ls, Lp+Ls) over [cached prefix | suffix]
        -> suffix_out (B, Ls, expert width)
        """
        streams = [(self.expert, suffix_embs, adarms_cond)]
        (suffix_out,), _ = self._run(streams, positions, mask, kv_cache=kv_cache)
        return suffix_out

    def _run(self, streams, positions, mask, kv_cache=None, return_cache=False):
        """
        Run 1 or 2 streams (model, tokens, cond) jointly through all layers.
        Called by forward_both (VLM + expert), forward_prefix (VLM, return_cache=True)
        and forward_suffix (expert, with kv_cache).
        positions: (B, L_new) for the new tokens of all streams, in sequence order.
        mask: (B, L_new, L_cached + L_new); cached keys come first.
        """

        models = [model for model, _, _ in streams]
        xs = [x for _, x, _ in streams]
        conds = [cond for _, _, cond in streams]

        # RoPE angles for the new tokens: computed once, used by every layer
        cfg = models[0].config
        cos, sin = rope_cos_sin(positions, cfg.head_dim, cfg.rope_theta)

        new_cache = []
        for i in range(cfg.depth):
            layers = [model.layers[i] for model in models]
            cached = kv_cache[i] if kv_cache is not None else None
            if self.gradient_checkpointing and self.training:
                xs, kv = checkpoint(
                    self._layer, layers, xs, conds, cos, sin, mask, cached, use_reentrant=False
                )
            else:
                xs, kv = self._layer(layers, xs, conds, cos, sin, mask, cached)

            if return_cache:
                new_cache.append(kv)

        outs = [model.norm(x, cond)[0] for model, x, cond in zip(models, xs, conds)]
        return outs, (new_cache if return_cache else None)

    @staticmethod
    def _layer(layers, xs, conds, cos, sin, mask, cached):
        """
        One layer for all streams: own pre_attn, joint attention, own post_attn.
        Used by all three modes; only the inputs differ:

                                 (layers/xs/conds) new tokens &
        mode                      streams          cos/sin,           mask             cached
        forward        (train)    VLM + expert     prefix + suffix    (Lp+Ls, Lp+Ls)   None
        forward_prefix (infer 1)  VLM              prefix             (Lp, Lp)         None
        forward_suffix (infer 2)  expert           suffix             (Ls, Lp+Ls)      prefix K/V

        Returns the updated tokens of each stream, and (k, v) of the new tokens
        (stored as the cache in forward_prefix; ignored otherwise).
        """

        # each stream: own RMSNorm/AdaRMSNorm + Q/K/V projections, split into heads
        qs, ks, vs, gates = [], [], [], []
        for layer, x, cond in zip(layers, xs, conds):
            q, k, v, gate = layer.pre_attn(x, cond)
            qs.append(q)
            ks.append(k)
            vs.append(v)
            gates.append(gate)

        # join the streams' Q/K/V along the token axis (a single stream is unchanged),
        # then apply RoPE to the new tokens
        q = torch.cat(qs, dim=2)  # (B, H, L_new, head_dim)
        k = torch.cat(ks, dim=2)  # (B, KV, L_new, head_dim)
        v = torch.cat(vs, dim=2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        new_kv = (k, v)   # what a cache stores for these tokens

        # Inference only: put cached keys/values of prefix in front of K/V of suffix
        if cached is not None:
            cached_k, cached_v = cached
            k = torch.cat([cached_k, k], dim=2)  # (B, KV, L_cached + L_new, head_dim)
            v = torch.cat([cached_v, v], dim=2)

        # attention on: joint Q/K/V or prefix Q/K/V or suffix Q/(cached_k | K)/(cached_v | V)
        out = attention(q, k, v, mask)            # (B, L_new, H*head_dim)

        # Give each stream back its own tokens and finish its layer
        new_xs = []
        start = 0
        for layer, x, gate, cond in zip(layers, xs, gates, conds):
            end = start + x.shape[1]  # this stream's token range

            # o_proj + gated residual 1 + post attn layer norm + MLP + gated residual 2
            new_xs.append(layer.post_attn(x, out[:, start:end], gate, cond))

            start = end

        return new_xs, new_kv
