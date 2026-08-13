"""
Image encoder module — re-exports original SAM1 components.
"""
from segment_anything.modeling.image_encoder import (
    ImageEncoderViT,
    Block,
    Attention,
    PatchEmbed,
    window_partition,
    window_unpartition,
    add_decomposed_rel_pos,
    get_rel_pos,
)
from segment_anything.modeling.common import LayerNorm2d
