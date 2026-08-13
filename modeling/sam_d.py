import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from modeling.drlora_block import DRLoRABlock
from modeling.decoder import Decoder3D, MLADecoder3D
from modeling.lora import LoRALinear


def interpolate_pos_embed(pos_embed: torch.Tensor, new_size: int) -> torch.Tensor:
    """Interpolate SAM positional embedding to a new spatial size.

    Args:
        pos_embed: (1, H, W, C) — original pos_embed (e.g., 1×64×64×768)
        new_size: target spatial size (e.g., 32 for 512×512 input with patch_size=16)
    Returns:
        (1, new_size, new_size, C)
    """
    # BHWC -> BCHW for interpolation
    x = pos_embed.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(new_size, new_size), mode='bicubic', align_corners=False)
    return x.permute(0, 2, 3, 1)  # back to BHWC


class SAM_D(nn.Module):
    """SAM1 ViT-B with DRLoRA + DSM for 3D medical image segmentation.

    Processes 3D volumes slice-by-slice through a frozen SAM1 encoder with:
    - DSM (Depth Shift Module) for cross-slice feature exchange
    - DRLoRA (Mixture of LoRA Experts) for slice-position-aware adaptation
    - Decoder: Decoder3D (conv3d), MLADecoder3D (mla), or SAM prompt decoder (sam)
    """

    def __init__(
        self,
        sam_model: nn.Module,
        num_slices: int = 32,
        slice_size: int = 512,
        rank: int = 16,
        num_experts: int = 4,
        shift_ratio: float = 0.25,
        num_classes: int = 2,
        decoder_type: str = "conv3d",
        no_z_embed: bool = False,
        no_decoder_lora: bool = False,
        routing_mode: str = "depth",
    ):
        super().__init__()
        encoder = sam_model.image_encoder
        embed_dim = encoder.pos_embed.shape[-1]  # 768 for ViT-B
        self.routing_mode = routing_mode

        # SAM components (will be frozen)
        self.patch_embed = encoder.patch_embed
        self.neck = encoder.neck

        # Interpolate pos_embed to match slice_size
        feat_size = slice_size // 16  # 512//16 = 32
        self.register_buffer(
            'pos_embed',
            interpolate_pos_embed(encoder.pos_embed, feat_size)
        )

        # Wrap each SAM block with DRLoRA
        self.blocks = nn.ModuleList([
            DRLoRABlock(block, embed_dim, rank, num_experts, shift_ratio, routing_mode=routing_mode)
            for block in encoder.blocks
        ])

        # Decoder
        self.decoder_type = decoder_type
        out_chans = 256  # SAM neck output channels

        if decoder_type == "mla":
            # Capture features after blocks 3, 6, 9, 12 (0-indexed: 2, 5, 8, 11)
            self.mla_feat_indices = {2, 5, 8, 11}
            # 4 neck_3d projections: embed_dim (768) -> 256
            self.neck_3d = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(embed_dim, out_chans, kernel_size=1, bias=False),
                    nn.InstanceNorm3d(out_chans),
                    nn.ReLU(),
                )
                for _ in range(4)
            ])
            self.decoder = MLADecoder3D(num_classes=num_classes)
        elif decoder_type == "conv3d":
            self.decoder = Decoder3D(out_chans, num_classes)
        elif decoder_type == "sam":
            # SAM prompt encoder (frozen) + mask decoder (frozen + LoRA)
            self.prompt_encoder = sam_model.prompt_encoder
            self.mask_decoder = sam_model.mask_decoder
            # Override spatial sizes: 1024→512 input, 64→32 features
            self.prompt_encoder.image_embedding_size = (feat_size, feat_size)
            self.prompt_encoder.input_image_size = (slice_size, slice_size)
            # Freeze prompt encoder
            for p in self.prompt_encoder.parameters():
                p.requires_grad = False
            # Freeze mask decoder, then optionally inject LoRA on Q/V
            for p in self.mask_decoder.parameters():
                p.requires_grad = False
            self.no_decoder_lora = no_decoder_lora
            if not no_decoder_lora:
                self._inject_decoder_lora(rank)
                # Unfreeze layer norms in mask decoder transformer
                for layer in self.mask_decoder.transformer.layers:
                    for norm in [layer.norm1, layer.norm2, layer.norm3, layer.norm4]:
                        for p in norm.parameters():
                            p.requires_grad = True
            # z-position embedding (3D prompt awareness)
            self.no_z_embed = no_z_embed
            if not no_z_embed:
                self.z_embed = nn.Sequential(
                    nn.Linear(1, 256), nn.GELU(), nn.Linear(256, 256),
                )
            # 3D refinement head: per-slice masks → multi-class 3D volume
            self.refine_3d = nn.Sequential(
                nn.Conv3d(1, 32, 3, padding=1),
                nn.InstanceNorm3d(32),
                nn.ReLU(),
                nn.Conv3d(32, num_classes, 1),
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}. Choose 'conv3d', 'mla', or 'sam'.")

        self.num_slices = num_slices
        self.slice_size = slice_size
        self.use_grad_checkpoint = True

        # Freeze SAM parameters, keep only DRLoRA + LayerNorms + decoder trainable
        self._freeze_sam_params()

    def _freeze_sam_params(self):
        """Freeze SAM backbone: patch_embed, neck, attention weights, MLP.
        Keep trainable: LayerNorms (norm1/norm2 in blocks), DRLoRA, decoder.
        """
        # Freeze patch_embed
        for p in self.patch_embed.parameters():
            p.requires_grad = False

        # Freeze neck
        for p in self.neck.parameters():
            p.requires_grad = False

        # Freeze attention and MLP weights in each block
        for drlora_block in self.blocks:
            block = drlora_block.block
            # Freeze attention: qkv, proj, rel_pos
            for p in block.attn.qkv.parameters():
                p.requires_grad = False
            for p in block.attn.proj.parameters():
                p.requires_grad = False
            if hasattr(block.attn, 'rel_pos_h'):
                block.attn.rel_pos_h.requires_grad = False
            if hasattr(block.attn, 'rel_pos_w'):
                block.attn.rel_pos_w.requires_grad = False
            # Freeze MLP
            for p in block.mlp.parameters():
                p.requires_grad = False
            # norm1 and norm2 remain trainable (default requires_grad=True)

    def _inject_decoder_lora(self, rank):
        """Replace q_proj and v_proj with LoRALinear in all mask decoder attention layers."""
        transformer = self.mask_decoder.transformer
        for layer in transformer.layers:
            for attn_name in ['self_attn', 'cross_attn_token_to_image', 'cross_attn_image_to_token']:
                attn = getattr(layer, attn_name)
                attn.q_proj = LoRALinear(attn.q_proj, rank)
                attn.v_proj = LoRALinear(attn.v_proj, rank)
        fa = transformer.final_attn_token_to_image
        fa.q_proj = LoRALinear(fa.q_proj, rank)
        fa.v_proj = LoRALinear(fa.v_proj, rank)

    def _decode_masks_batched(self, image_embeddings, image_pe, sparse_emb, dense_emb):
        """Batched SAM mask decode: 1 prompt per image (no repeat_interleave).

        SAM's original predict_masks uses repeat_interleave which causes quadratic
        memory when batch > 1. This method handles 1:1 image-to-prompt mapping.

        Args:
            image_embeddings: (N, 256, fH, fW) — encoder features after neck
            image_pe: (1, 256, fH, fW) — positional encoding from prompt encoder
            sparse_emb: (N, n_tokens, 256) — sparse prompt embeddings
            dense_emb: (N, 256, fH, fW) — dense prompt embeddings + z_embed
        Returns:
            masks: (N, 1, 4*fH, 4*fW) — single-mask logits
        """
        md = self.mask_decoder
        # Prepare output tokens: iou_token + mask_tokens
        output_tokens = torch.cat([md.iou_token.weight, md.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_emb.size(0), -1, -1)
        tokens = torch.cat([output_tokens, sparse_emb], dim=1)

        # 1:1 mapping — NOT repeat_interleave
        src = image_embeddings + dense_emb
        pos_src = image_pe.expand(src.shape[0], -1, -1, -1)
        b, c, h, w = src.shape

        # TwoWayTransformer
        hs, src = md.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 1:2, :]  # mask token 0 only

        # Upscale image embeddings
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled = md.output_upscaling(src)  # (N, 32, 4*fH, 4*fW)

        # Hypernetwork for mask 0
        hyper = md.output_hypernetworks_mlps[0](mask_tokens_out[:, 0, :])  # (N, 32)
        b2, c2, h2, w2 = upscaled.shape
        masks = (hyper.unsqueeze(1) @ upscaled.view(b2, c2, h2 * w2)).view(b2, 1, h2, w2)
        return masks

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Subsample depth and resize spatial dims for SAM processing.

        Args:
            image: (B, 3, D, H, W)
        Returns:
            (B*num_slices, 3, slice_size, slice_size)
        """
        B, C, D, H, W = image.shape

        # Uniform depth subsampling
        indices = torch.linspace(0, D - 1, self.num_slices, device=image.device).long()
        x = image[:, :, indices]  # (B, 3, num_slices, H, W)

        # Reshape to per-slice batch
        x = x.permute(0, 2, 1, 3, 4).reshape(B * self.num_slices, C, H, W)

        # Resize spatial dims
        if H != self.slice_size or W != self.slice_size:
            x = F.interpolate(x, size=(self.slice_size, self.slice_size),
                              mode='bilinear', align_corners=False)
        return x

    def forward(self, image: torch.Tensor, point_coords=None, point_labels=None) -> torch.Tensor:
        """
        Args:
            image: (B, 3, D, H, W) — 3D volume from dataset
            point_coords: (B, N, 3) — (z, h, w) in crop/volume space (SAM decoder only)
            point_labels: (B, N) — 1=positive, 0=negative (SAM decoder only)
        Returns:
            (B, num_classes, D, H, W) — segmentation logits matching input size
        """
        B, C, D_orig, H_orig, W_orig = image.shape

        # 1. Preprocess: subsample depth, resize spatial
        x = self._preprocess(image)  # (B*D, 3, 512, 512)

        # 2. Patch embed + positional embedding
        x = self.patch_embed(x)  # (B*D, feat_h, feat_w, embed_dim) BHWC
        x = x + self.pos_embed   # broadcast (1, feat_h, feat_w, embed_dim)

        # 3. Slice positions for DRLoRA routing
        D = self.num_slices
        slice_pos = torch.linspace(0, 1, D, device=x.device).unsqueeze(-1)  # (D, 1)
        if B > 1:
            slice_pos = slice_pos.repeat(B, 1)  # (B*D, 1)

        # 4. Run through DRLoRA blocks, optionally capturing intermediates for MLA
        capture_mla = (self.decoder_type == "mla")
        intermediate_features = []
        feat_idx = 0
        aux_losses = []

        for idx, block in enumerate(self.blocks):
            if self.use_grad_checkpoint and self.training:
                x, aux = grad_checkpoint(block, x, slice_pos, use_reentrant=False)
            else:
                x, aux = block(x, slice_pos)
            aux_losses.append(aux)

            if capture_mla and idx in self.mla_feat_indices:
                # x: (B*D, fH, fW, 768) BHWC -> (B*D, 768, fH, fW) -> (B, 768, D, fH, fW)
                feat = x.permute(0, 3, 1, 2)
                fH, fW = x.shape[1], x.shape[2]
                feat_3d = feat.view(B, D, -1, fH, fW).permute(0, 2, 1, 3, 4)
                projected = self.neck_3d[feat_idx](feat_3d)  # (B, 256, D, fH, fW)
                intermediate_features.append(projected)
                feat_idx += 1

        target_size = (D_orig, H_orig, W_orig)

        if capture_mla:
            # 5a. MLA path: use multi-scale features
            # img_resize: grayscale input resized to target_size
            img_resize = F.interpolate(
                image.mean(dim=1, keepdim=True), size=target_size,
                mode='trilinear', align_corners=False,
            )  # (B, 1, D_orig, H_orig, W_orig)
            logits = self.decoder(intermediate_features, img_resize, target_size)
        elif self.decoder_type == "sam":
            # 5c. SAM prompt decoder path
            logits = self._forward_sam_decoder(
                x, image, B, D, D_orig, H_orig, W_orig,
                point_coords, point_labels, target_size,
            )
        else:
            # 5b. Conv3d path: original behavior
            x = self.neck(x.permute(0, 3, 1, 2))  # (B*D, 256, fH, fW)
            _, C_out, fH, fW = x.shape
            x = x.view(B, D, C_out, fH, fW).permute(0, 2, 1, 3, 4)  # (B, 256, D, fH, fW)
            logits = self.decoder(x, target_size)

        if self.training:
            # mean aux loss across the 24 DRLoRA modules (12 blocks × Q+V)
            total_aux = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())
            return logits, total_aux
        return logits

    def _forward_sam_decoder(
        self, x, image, B, D, D_orig, H_orig, W_orig,
        point_coords, point_labels, target_size,
    ):
        """SAM prompt-based decoder path.

        Args:
            x: (B*D, fH, fW, 768) — encoder output in BHWC format
            image: (B, 3, D_orig, H_orig, W_orig) — original input (unused here)
            B, D: batch size and num_slices
            D_orig, H_orig, W_orig: original volume dimensions
            point_coords: (B, N, 3) — (z, h, w) in crop space
            point_labels: (B, N) — 1=positive, 0=negative
            target_size: (D_orig, H_orig, W_orig)
        Returns:
            (B, num_classes, D_orig, H_orig, W_orig)
        """
        # SAM neck: BHWC → BCHW → (B*D, 256, fH, fW)
        x = self.neck(x.permute(0, 3, 1, 2))
        _, C_neck, fH, fW = x.shape

        # Coordinate conversion: (z, h, w) in crop space → SAM (x, y) format
        scale_h = self.slice_size / H_orig
        scale_w = self.slice_size / W_orig
        sam_points_xy = torch.stack([
            point_coords[:, :, 2].float() * scale_w,   # x = w * scale
            point_coords[:, :, 1].float() * scale_h,   # y = h * scale
        ], dim=-1)  # (B, N_pts, 2) in SAM (x, y) format

        # SAM prompt encoder (frozen)
        sparse_emb, dense_emb = self.prompt_encoder(
            points=(sam_points_xy, point_labels),
            boxes=None,
            masks=None,
        )
        # sparse_emb: (B, N_tokens, 256), dense_emb: (B, 256, fH, fW)

        # Expand prompts from (B, ...) to (B*D, ...)
        dense_exp = dense_emb[:, None].expand(-1, D, -1, -1, -1).reshape(B * D, C_neck, fH, fW)

        # z-position encoding: each slice gets relative distance to point(s)
        if not self.no_z_embed:
            z_point = point_coords[:, :, 0].float().mean(dim=1)  # (B,) — mean z
            slice_indices = torch.linspace(0, D - 1, D, device=x.device)  # (D,)
            z_rel = (z_point[:, None] - slice_indices[None, :]) / D  # (B, D) normalized
            z_emb = self.z_embed(z_rel.reshape(B * D, 1))  # (B*D, 256)
            dense_exp = dense_exp + z_emb[:, :, None, None]  # add z-position
        sparse_exp = sparse_emb[:, None].expand(-1, D, -1, -1).reshape(B * D, -1, 256)

        # Batched mask decode (fixes SAM's repeat_interleave bug)
        image_pe = self.prompt_encoder.get_dense_pe()  # (1, 256, fH, fW)
        masks = self._decode_masks_batched(x, image_pe, sparse_exp, dense_exp)
        # masks: (B*D, 1, 4*fH, 4*fW) e.g. (B*32, 1, 128, 128)

        # Reshape to 3D volume
        mask_h, mask_w = masks.shape[-2], masks.shape[-1]
        masks = masks.view(B, D, 1, mask_h, mask_w).permute(0, 2, 1, 3, 4)
        # (B, 1, D, mask_h, mask_w)

        # 3D refinement: cross-slice smoothing + multi-class output
        logits = self.refine_3d(masks)  # (B, num_classes, D, mask_h, mask_w)

        # Interpolate to target size
        logits = F.interpolate(logits, size=target_size, mode='trilinear', align_corners=False)
        return logits
