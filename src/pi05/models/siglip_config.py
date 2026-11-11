from dataclasses import dataclass

@dataclass(frozen=True)
class SiglipConfig:
    image_size: int = 224                # input resolution (square)
    patch_size: int = 14                 # 224/14 = 16x16 grid = 256 tokens per image
    width: int = 1152                    # size of each patch token
    depth: int = 27                      # number of encoder layers
    mlp_dim: int = 4304                  # hidden size inside MLP
    num_heads: int = 16                  # head_dim = 1152/16 = 72
    layer_norm_eps: float = 1e-6         # LayerNorm divide-by-zero guard

    # "eager": plain PyTorch ops, readable and exactly reproducible
    # "sdpa":  fused PyTorch kernel, faster and uses less memory
    attn_implementation: str = "eager"


def get_siglip_config(variant: str) -> SiglipConfig:
    variants = {
        "so400m": SiglipConfig(),  # SigLIP So400m/14, PaliGemma's vision tower
        
        "siglip_tiny": SiglipConfig(
            image_size=28, patch_size=14, width=24, depth=2, mlp_dim=48, num_heads=4
        ),
        
         # tiny, but real 224 images fit (dev runs)
        "siglip_tiny_224": SiglipConfig( 
            image_size=224, patch_size=14, width=24, depth=2, mlp_dim=48, num_heads=4,
        ),
    }
    return variants[variant]