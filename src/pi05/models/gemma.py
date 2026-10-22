import torch
import torch.nn.functional as F
from torch import nn
from .gemma_config import GemmaConfig


# Additive attention-mask value for "cannot attend". Large finite negative instead of -inf.
# fits in bf16, and fully masked rows (padding) give a uniform softmax instead of NaN.
MASK_VALUE = -2.3819763e38


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float):
    """
    positions (B, L)

    inv_freq[i] = 1 / theta^(2i/head_dim); angle = position * inv_freq[i]

    Note: torch.arange(0, 256, 2) / 256 == torch.arange(128) / 128  bec (2i/256 = i/128)
    """
    i = torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32)
    inv_freq = 1.0 / theta ** (i / head_dim)                   # (head_dim/2,)
    freqs = positions.float()[..., None]*inv_freq            # (B, L, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                  # (B, L, head_dim)
    return emb.cos(), emb.sin()

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x       : (B, H, L, head_dim)
    cos/sin : (B, L, head_dim)
    returns rotated x, same shape and dtype.

    Example, head_dim = 6 (pairs: a1-b1, a2-b2, a3-b3):
        x       = [ a1   a2   a3   b1  b2  b3]
        rotated = [-b1  -b2  -b3   a1  a2  a3]
        cos     = [ c1   c2   c3   c1  c2  c3]     ci = cos(position * inv_freq[i])
        sin     = [ s1   s2   s3   s1  s2  s3]     si = sin(position * inv_freq[i])
        out     = x * cos + rotated * sin           (elementwise)

    Per pair this is a 2D rotation:
        a1' = a1*c1 - b1*s1
        b1' = b1*c1 + a1*s1       i.e.  [a1', b1'] = [[c1, -s1], [s1, c1]] @ [a1, b1]
    """
    cos = cos[:, None].to(x.dtype)                 # (B, 1, L, head_dim)
    sin = sin[:, None].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)                    # first half, second half
    rotated = torch.cat([-x2, x1], dim=-1)         # rotate_half to pair i with i+half
    return x * cos + rotated * sin

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    q    : (B, H, L, head_dim)     H query heads, L tokens
    k, v : (B, KV, L, head_dim)    KV key/value heads (shared by the H query heads)
    mask : (B, L, L) bool          True = query i may attend key j
    returns (B, L, H * head_dim),  ready for o_proj
    """
    B, H, L, D = q.shape
    n_rep = H // k.shape[1]
    k = k.repeat_interleave(n_rep, dim=1)                    # (B, H, L, D): 
    v = v.repeat_interleave(n_rep, dim=1)

    scores = (q @ k.transpose(-1, -2)) * D ** -0.5           # (B, H, L, S)
    bias = torch.where(mask[:, None], 0.0, MASK_VALUE)       # (B, 1, L, S): same for all heads
    probs = torch.softmax(scores.float() + bias, dim=-1).to(q.dtype)
    out = probs @ v                                          # (B, H, L, D)
    return out.transpose(1, 2).reshape(B, L, H * D)          # (B, L, H*D)

def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    """
    x + y for plain blocks (VLM)
    x + y * gate for AdaRMS blocks (expert)
    """
    return x + y if gate is None else x + y * gate


class RMSNorm(nn.Module):
    """
    Gemma RMSNorm : plain/normal or adaptive

    rms  = sqrt(mean(x²) + eps)
    norm(x)   = x / rms

    plain:    out = norm(x) * (1 + weight)
    adaptive: out = norm(x) * (1 + scale) + shift, with scale/shift/gate from cond
    Returns (out, gate); gate is None in plain mode.
    """

    def __init__(self, dim: int, eps: float = 1e-6, adaptive: bool = False):
        super().__init__()
        self.eps = eps
        self.adaptive = adaptive
        if adaptive:
            self.dense = nn.Linear(dim, 3*dim)   # ada_cond -> scale, shift, gate
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        # x    : (B, L, dim)
        # cond : (B, dim) for adaptive only

        dtype = x.dtype
        x = x.float()                        # norm math in float32
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.adaptive:
            if cond is not None:
                raise ValueError("plain RMSNorm does not take cond")
            return (normed * (1 + self.weight.float())).to(dtype), None

        if cond is None:
            raise ValueError("adaptive RMSNorm requires cond")
        mod = self.dense(cond)                          # (B, 3*dim)
        if x.ndim == 3:
            mod = mod.unsqueeze(1)                      # (B, 1, 3*dim)
        scale, shift, gate = mod.float().chunk(3, dim=-1)
        out = normed * (1 + scale) + shift
        return out.to(dtype), gate.to(dtype)


class GemmaMLP(nn.Module):
    """
    GeGLU MLP: down(gelu_tanh(gate(x)) * up(x)). Same for VLM and expert.
    """
    def __init__(self, width: int, mlp_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, width) -> (B, L, width)
        gate = F.gelu(self.gate_proj(x), approximate="tanh") # (B, L, mlp_dim)
        return self.down_proj(gate * self.up_proj(x))


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.width, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.width, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.width, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.width, bias=False)

    def project_qkv(self, x: torch.Tensor):
        """
        x (B, L, D=width)
        RoPE not applied yet
        
        return
            q    (B, H, L, D)
            k, v (B, KV, L, D)
        """
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(self, x, cos, sin, mask):
        q, k, v = self.project_qkv(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return self.o_proj(attention(q, k, v, mask))


class GemmaBlock(nn.Module):
    """
    One transformer layer: norm -> attention -> residual, norm -> MLP -> residual.
    Split into pre_attn / post_attn so joint attention can run in between.
    """
    def __init__(self, config: GemmaConfig):
        super().__init__()
        ada = config.use_adarms
        self.input_layernorm = RMSNorm(config.width, config.rms_eps, adaptive=ada)
        self.self_attn = GemmaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.width, config.rms_eps, adaptive=ada)
        self.mlp = GemmaMLP(config.width, config.mlp_dim)

    def pre_attn(self, x, cond=None):
        """
        x (B, L, D)
        cond (B, D) if not None

        without RoPE and the attention

        returns 
            q   (B, H, L, D)
            k/v (B, KV, L, D)
        """
        h, gate = self.input_layernorm(x, cond)
        q, k, v = self.self_attn.project_qkv(h)
        return q, k, v, gate

    def post_attn(self, x, attn_out, gate, cond=None):
        x = gated_residual(x, self.self_attn.o_proj(attn_out), gate)  # residual 1
        h, gate = self.post_attention_layernorm(x, cond)
        return gated_residual(x, self.mlp(h), gate)                   # residual 2

    def forward(self, x, cos, sin, mask, cond=None):
        """
        single-stream layer: x (B, L, D) -> (B, L, D)
        """
        q, k, v, gate = self.pre_attn(x, cond)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return self.post_attn(x, attention(q, k, v, mask), gate, cond)


class GemmaModel(nn.Module):
    """Token embedding (optional) + stack of GemmaBlocks + final norm."""

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.width) if config.vocab_size else None
        self.layers = nn.ModuleList([GemmaBlock(config) for _ in range(config.depth)])
        self.norm = RMSNorm(config.width, config.rms_eps, adaptive=config.use_adarms)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Token ids (B, L) 
        
        return embeddings (B, L, D), scaled by sqrt(D)
        """
        if self.embed_tokens is None:
            raise ValueError("this model has no token embedding (action expert)")
        return self.embed_tokens(ids) * self.config.width ** 0.5

    def forward(self, embs, positions, mask, cond=None):
        """
        Single-stream forward.
        embs (B, L, D)
        positions (B, L)
        mask (B, L, L) 
        cond (B, D) or None

        return (B, L, D)
        """
        cfg = self.config
        cos, sin = rope_cos_sin(positions, cfg.head_dim, cfg.rope_theta) # once, shared by all layers
        x = embs
        for layer in self.layers:
            x = layer(x, cos, sin, mask, cond)
        x, _ = self.norm(x, cond)              # final norm; its gate is unused
        return x