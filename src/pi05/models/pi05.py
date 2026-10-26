import math

import torch
import torch.nn.functional as F
from torch import nn

from .gemma_config import get_gemma_config
from .observation import Observation
from .paligemma_with_expert import PaliGemmaWithExpert
from .pi05_config import Pi05Config
from .siglip_config import get_siglip_config


def sinusoidal_time_embedding(
    t: torch.Tensor, dim: int, min_period: float, max_period: float
) -> torch.Tensor:
    """
    t (B,) in [0, 1] -> (B, dim): [sin | cos] at dim/2 geometrically spaced periods.
    Computed in float64 for accuracy, returned in t's dtype.
    """
    if t.ndim != 1:
        raise ValueError(f"t must have shape (B,), got {tuple(t.shape)}")
    if dim % 2 != 0:
        raise ValueError(f"dim must be even (half sin, half cos), got {dim}")

    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64, device=t.device)
    period = min_period * (max_period / min_period) ** fraction
    angle = (2 * math.pi / period)[None, :] * t.to(torch.float64)[:, None]  # (B, dim/2)
    return torch.cat([angle.sin(), angle.cos()], dim=1).to(t.dtype)

def make_attention_mask(pad: torch.Tensor, block_start: torch.Tensor) -> torch.Tensor:
    """
    pad         (B, L) bool: True = real token, False = padding
    block_start (B, L) bool: True = this token starts a new attention block
    Token i may attend token j if j's block is not later than i's, and both are real.
    returns (B, L, L) bool mask for attention
    """
    block = torch.cumsum(block_start.long(), dim=1)   # block number of each token
    allowed = block[:, None, :] <= block[:, :, None]  # [b, i, j]: key j's block <= query i's
    both_real = pad[:, None, :] & pad[:, :, None]     # key real AND query real
    return allowed & both_real

def make_positions(pad: torch.Tensor) -> torch.Tensor:
    """
    RoPE positions counting only real tokens: pad [1,1,0,1] -> [0,1,1,2].
    """
    return torch.cumsum(pad.long(), dim=1) - 1

def sample_time(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Flow-matching timestep per sample, in [0.001, 1]. Beta(1.5, 1) favours t near 1 (noisier).
    """
    beta = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))
    return beta.sample((batch_size,)) * 0.999 + 0.001

def prefix_suffix_masks(prefix_pad: torch.Tensor, suffix_len: int):
    """
    pad and block_start for [prefix | suffix]: the prefix is one block; the suffix starts a new one.
    prefix_pad (B, Lp) -> pad (B, Lp+Ls), block_start (B, Lp+Ls)
    """
    B, Lp = prefix_pad.shape
    suffix_pad = torch.ones(B, suffix_len, dtype=torch.bool, device=prefix_pad.device)
    pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    block_start = torch.zeros_like(pad)
    block_start[:, Lp] = True
    return pad, block_start


class Pi05(nn.Module):
    """
    pi0.5: observations -> flow-matching loss (training) or actions (inference).
    """

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config
        self.paligemma_with_expert = PaliGemmaWithExpert(
            get_siglip_config(config.vision_variant),
            get_gemma_config(config.vlm_variant),
            get_gemma_config(config.expert_variant),
        )
        expert_width = self.paligemma_with_expert.expert.config.width
        self.action_in_proj = nn.Linear(config.action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, config.action_dim)
        self.time_mlp_in = nn.Linear(expert_width, expert_width)
        self.time_mlp_out = nn.Linear(expert_width, expert_width)

    def embed_prefix(self, obs: Observation):
        """
        obs -> prefix_embs (B, Lp, vlm width), prefix_pad (B, Lp) bool
        Lp = N_cams * patches_per_image + prompt length; order: camera 0, camera 1, ..., prompt
        """
        paligemma = self.paligemma_with_expert.paligemma
        B, n_cams = obs.image_masks.shape

        # all cameras of all samples through SigLIP in one call
        img_embs = paligemma.embed_image(obs.images.flatten(0, 1))  # (B*N, P, width)
        patches = img_embs.shape[1]
        img_embs = img_embs.reshape(B, n_cams * patches, -1)        # (B, N*P, width)
        img_pad = obs.image_masks.repeat_interleave(patches, dim=1) # (B, N*P)

        lang_embs = paligemma.embed_language(obs.prompt_ids)  # (B, L, width), scaled by sqrt(width)

        prefix_embs = torch.cat([img_embs, lang_embs], dim=1)
        prefix_pad = torch.cat([img_pad, obs.prompt_mask], dim=1)
        return prefix_embs, prefix_pad

    def embed_suffix(self, noisy_actions: torch.Tensor, t: torch.Tensor):
        """
        noisy_actions (B, horizon, action_dim), t (B,)
        -> action_embs (B, horizon, expert width), adarms_cond (B, expert width)
        """
        time_emb = sinusoidal_time_embedding(
            t,
            self.time_mlp_in.in_features,
            self.config.time_min_period,
            self.config.time_max_period,
        )
        time_hidden = F.silu(self.time_mlp_in(time_emb))
        adarms_cond = F.silu(self.time_mlp_out(time_hidden))
        action_embs = self.action_in_proj(noisy_actions)
        return action_embs, adarms_cond

    def forward(self, obs: Observation, actions: torch.Tensor, noise=None, t=None) -> torch.Tensor:
        """
        Training: per-element flow-matching loss.
        obs: preprocessed observation (images resized to 224 x 224, in [-1, 1])
        actions (B, horizon, action_dim): clean target actions
        noise, t: optional fixed values (tests, parity); sampled if None
        returns loss (B, horizon, action_dim), float32; the training loop takes the mean
        """
        B, horizon, _ = actions.shape
        if noise is None:
            noise = torch.randn_like(actions)
        if t is None:
            t = sample_time(B, actions.device)

        t_expanded = t[:, None, None]    # (B, 1, 1): broadcast over horizon and dims
        x_t = t_expanded * noise + (1-t_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad = self.embed_prefix(obs)
        suffix_embs, adarms_cond = self.embed_suffix(x_t, t)
        pad, block_start = prefix_suffix_masks(prefix_pad, horizon)
        mask = make_attention_mask(pad, block_start)
        positions = make_positions(pad)

        _, suffix_out = self.paligemma_with_expert(prefix_embs, suffix_embs, positions, mask, adarms_cond)
        v_t = self.action_out_proj(suffix_out)  # predicted velocity (B, horizon, action_dim)
        return F.mse_loss(v_t.float(), u_t.float(), reduction="none")

    def set_gradient_checkpointing(self, enabled: bool):
        """
        Training memory saver; delegates to the transformer part (the only part where it pays off).
        """
        self.paligemma_with_expert.set_gradient_checkpointing(enabled)

    def sample_actions(self, obs: Observation, noise=None, num_steps=None) -> torch.Tensor:
        """
        Inference: denoise from pure noise (t=1) to actions (t=0) with Euler steps.
        obs: preprocessed observation
        noise: optional fixed starting noise (B, horizon, action_dim); sampled if None
        num_steps: Euler steps (default: config.num_denoise_steps)
        returns actions (B, horizon, action_dim)
        """
        cfg = self.config
        num_steps = num_steps or cfg.num_denoise_steps
        B, device = obs.prompt_ids.shape[0], obs.prompt_ids.device
        if noise is None:
            noise = torch.randn(B, cfg.action_horizon, cfg.action_dim, device=device)

        # full-sequence mask and positions, exactly as in training; prefix/suffix parts are slices
        prefix_embs, prefix_pad = self.embed_prefix(obs)
        Lp = prefix_pad.shape[1]
        pad, block_start = prefix_suffix_masks(prefix_pad, cfg.action_horizon)
        mask = make_attention_mask(pad, block_start)
        positions = make_positions(pad)

        # prefix once -> KV cache
        _, kv_cache = self.paligemma_with_expert.forward_prefix(
            prefix_embs, positions[:, :Lp], mask[:, :Lp, :Lp]
        )

        # Euler steps from t = 1 (noise) to t = 0 (actions)
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = torch.full((B,), 1.0 + step * dt, device=device)
            suffix_embs, adarms_cond = self.embed_suffix(x_t, t)
            suffix_out = self.paligemma_with_expert.forward_suffix(
                suffix_embs, positions[:, Lp:], mask[:, Lp:, :], adarms_cond, kv_cache
            )
            v_t = self.action_out_proj(suffix_out)
            x_t = x_t + dt * v_t
        return x_t