import torch
import torch.nn as nn

from modeling.image_encoder import window_partition, window_unpartition, add_decomposed_rel_pos
from modeling.lora import DRLoRA
from modeling.dsm import depth_shift


class DRLoRABlock(nn.Module):
    """Wraps a frozen SAM1 Block, injecting DSM and DRLoRA on Q and V.

    forward returns (x, aux_loss). aux_loss is the sum of Q+V routing aux losses
    (zero for routing modes that don't use one).
    """

    def __init__(
        self,
        sam_block: nn.Module,
        embed_dim: int = 768,
        rank: int = 16,
        num_experts: int = 4,
        shift_ratio: float = 0.25,
        routing_mode: str = "depth",
    ):
        super().__init__()
        self.block = sam_block
        self.drlora_q = DRLoRA(embed_dim, rank, num_experts, routing_mode=routing_mode)
        self.drlora_v = DRLoRA(embed_dim, rank, num_experts, routing_mode=routing_mode)
        self.shift_ratio = shift_ratio

    def forward(self, x: torch.Tensor, slice_positions: torch.Tensor):
        """
        Args:
            x: (D, H, W, C)
            slice_positions: (D, 1)
        Returns:
            (x_out, aux_loss)
        """
        x = depth_shift(x, self.shift_ratio)

        shortcut = x
        x = self.block.norm1(x)

        if self.block.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.block.window_size)
            Hp, Wp = pad_hw
            num_windows = (Hp // self.block.window_size) * (Wp // self.block.window_size)
            sp = slice_positions.repeat_interleave(num_windows, dim=0)
        else:
            sp = slice_positions

        x, aux_loss = self._attention_with_drlora(x, sp)

        if self.block.window_size > 0:
            x = window_unpartition(x, self.block.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.block.mlp(self.block.norm2(x))
        return x, aux_loss

    def _attention_with_drlora(self, x: torch.Tensor, sp: torch.Tensor):
        attn = self.block.attn
        B, H, W, _ = x.shape
        dim = attn.qkv.in_features

        qkv = attn.qkv(x)

        q_delta, aux_q = self.drlora_q(x, sp)
        v_delta, aux_v = self.drlora_v(x, sp)

        qkv = torch.cat([
            qkv[..., :dim] + q_delta,
            qkv[..., dim:2 * dim],
            qkv[..., 2 * dim:] + v_delta,
        ], dim=-1)

        qkv = qkv.reshape(B, H * W, 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * attn.num_heads, H * W, -1).unbind(0)

        attn_weights = (q * attn.scale) @ k.transpose(-2, -1)

        if attn.use_rel_pos:
            attn_weights = add_decomposed_rel_pos(
                attn_weights, q, attn.rel_pos_h, attn.rel_pos_w, (H, W), (H, W)
            )

        attn_weights = attn_weights.softmax(dim=-1)
        x = (attn_weights @ v).view(B, attn.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = attn.proj(x)

        return x, aux_q + aux_v
