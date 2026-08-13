import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder3D(nn.Module):
    """3D convolutional decoder that upsamples SAM features to segmentation masks.

    Takes reassembled 3D feature volume (B, 256, D, H', W') and upsamples
    through 2 ConvTranspose3d stages, then interpolates to match target size.

    Shape flow (for default 32×32×32 feature volume):
        (B, 256, 32, 32, 32) → up1 → (B, 128, 64, 64, 64)
                              → up2 → (B, 64, 128, 128, 128)
                              → head → (B, num_classes, 128, 128, 128)
                              → interpolate to target size
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 2):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 128, kernel_size=2, stride=2),
            nn.InstanceNorm3d(128),
            nn.GELU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.InstanceNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.GELU(),
        )
        self.head = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        Args:
            x: (B, 256, D, H', W') — 3D feature volume from encoder
            target_size: (D_out, H_out, W_out) — desired output spatial size
        Returns:
            (B, num_classes, D_out, H_out, W_out)
        """
        x = self.up1(x)
        x = self.up2(x)
        x = self.head(x)
        if tuple(x.shape[2:]) != tuple(target_size):
            x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
        return x


class MLAHead(nn.Module):
    """Multi-Layer Aggregation head with 4 parallel Conv3d upsampling pathways.

    Each pathway processes one feature level (B, 256, D, H, W) through two
    Conv3d layers, then upsamples to output_size. All 4 outputs are concatenated.

    Output: (B, 4*mla_channels, D', H', W')
    """

    def __init__(self, in_channels: int = 256, mla_channels: int = 128):
        super().__init__()
        for i in range(4):
            setattr(self, f"head{i}", nn.Sequential(
                nn.Conv3d(in_channels, mla_channels, 3, padding=1, bias=False),
                nn.InstanceNorm3d(mla_channels),
                nn.ReLU(),
                nn.Conv3d(mla_channels, mla_channels, 3, padding=1, bias=False),
                nn.InstanceNorm3d(mla_channels),
                nn.ReLU(),
            ))

    def forward(self, features: list, output_size: tuple) -> torch.Tensor:
        """
        Args:
            features: list of 4 tensors, each (B, 256, D, H, W)
            output_size: (D', H', W') target spatial size after upsampling
        Returns:
            (B, 4*mla_channels, D', H', W')
        """
        outputs = []
        for i in range(4):
            head = getattr(self, f"head{i}")
            x = head(features[i])
            x = F.interpolate(x, size=output_size, mode='trilinear', align_corners=False)
            outputs.append(x)
        return torch.cat(outputs, dim=1)


class MLADecoder3D(nn.Module):
    """MLA decoder: 4 multi-scale features + resized image -> segmentation logits.

    Combines MLAHead output (B, 512, D', H', W') with a single-channel resized
    image (B, 1, D', H', W'), then classifies via Conv3d layers.

    Shape flow (for 4 features at 32x32x32, target 128x128x128):
        4 x (B, 256, 32, 32, 32) -> MLAHead -> (B, 512, 128, 128, 128)
        cat with img_resize      -> (B, 513, 128, 128, 128)
        cls_head                 -> (B, num_classes, 128, 128, 128)
    """

    def __init__(self, num_classes: int = 2, mla_channels: int = 128):
        super().__init__()
        self.mla_head = MLAHead(in_channels=256, mla_channels=mla_channels)
        self.cls_head = nn.Sequential(
            nn.Conv3d(4 * mla_channels + 1, mla_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(mla_channels),
            nn.ReLU(),
            nn.Conv3d(mla_channels, num_classes, 1),
        )

    def forward(
        self,
        features: list,
        img_resize: torch.Tensor,
        target_size: tuple,
    ) -> torch.Tensor:
        """
        Args:
            features: list of 4 tensors, each (B, 256, D, H, W)
            img_resize: (B, 1, D', H', W') single-channel resized input image
            target_size: (D_out, H_out, W_out) final output spatial size
        Returns:
            (B, num_classes, D_out, H_out, W_out)
        """
        x = self.mla_head(features, target_size)  # (B, 512, D', H', W')
        x = torch.cat([x, img_resize], dim=1)     # (B, 513, D', H', W')
        x = self.cls_head(x)                       # (B, num_classes, D', H', W')
        if tuple(x.shape[2:]) != tuple(target_size):
            x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
        return x
