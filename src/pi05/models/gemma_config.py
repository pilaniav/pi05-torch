from dataclasses import dataclass

@dataclass(frozen=True)
class GemmaConfig:
    width: int                         # token's vector size
    depth: int                         # num of transformer blocks/layer
    mlp_dim: int                       # hidden size inside MLP
    num_heads: int = 8                 # num of query heads in attention
    num_kv_heads: int = 1              # num of key/value heads, shared by queries
    head_dim: int = 256                # size of each Q/K/V head
    vocab_size: int | None = 257152    # token table size; None = no embedding table
    use_adarms: bool = False           # timestep-conditioned norms (expert only)
    rms_eps: float = 1e-6              # RMSNorm divide-by-zero guard
    rope_theta: float = 10_000.0       # RoPE base frequency

def get_gemma_config(variant: str) -> GemmaConfig:
    variants = {
        # Language model in Paligemma
        "gemma_2b": GemmaConfig(width=2048, depth=18, mlp_dim=16384),
        # Pi0.5 action expert
        "gemma_300m": GemmaConfig(
            width=1024, depth=18, mlp_dim=4096, vocab_size=None, use_adarms=True
        ),
        
        # tiny models for CPU unit tests
        "gemma_tiny": GemmaConfig(
            width=32, depth=2, mlp_dim=64, num_heads=4, num_kv_heads=1, head_dim=8, vocab_size=100
        ),
        "gemma_tiny_expert": GemmaConfig(
            width=16,
            depth=2,
            mlp_dim=32,
            num_heads=4,
            num_kv_heads=1,
            head_dim=8,
            vocab_size=None,
            use_adarms=True,
        ),
    }
    return variants[variant]


