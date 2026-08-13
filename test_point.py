"""Test with multiple 128x128x128 crops centered near the tumor, averaged.

For each test case:
1. Find the center of mass of the ground-truth tumor label
2. Sample N crops with different random jitters around the center
3. Run the model on each crop, accumulate softmax probabilities on the full volume
4. Average overlapping regions, argmax, then post-process (keep largest component)
5. Compute Dice / NSD on the full volume prediction
"""
from dataset.datasets import load_data_volume, create_dataset
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from modeling.sam_d import SAM_D
import argparse
import numpy as np
import logging
import torch
import torch.nn.functional as F
import os
import nibabel as nib
from utils.util import setup_logger, migrate_legacy_state_dict
import surface_distance
from surface_distance import metrics
from scipy import ndimage


def compute_dice(pred, label):
    pred = pred.float()
    label = label.float()
    intersection = (pred * label).sum()
    union = pred.sum() + label.sum()
    if union == 0:
        return 1.0
    return (2.0 * intersection / union).item()


def compute_nsd(pred, label, spacing, tolerance=5):
    pred_np = pred.cpu().numpy().astype(bool)
    label_np = label.cpu().numpy().astype(bool)
    if not pred_np.any() and not label_np.any():
        return 1.0
    if not pred_np.any() or not label_np.any():
        return 0.0
    sd = surface_distance.compute_surface_distances(label_np, pred_np, spacing_mm=spacing)
    return metrics.compute_surface_dice_at_tolerance(sd, tolerance)


def keep_largest_components(mask_np, num_keep=1, min_size=100):
    """Keep the largest connected component(s), remove small noise."""
    if not mask_np.any():
        return mask_np
    labeled, num_features = ndimage.label(mask_np)
    if num_features <= num_keep and min_size == 0:
        return mask_np
    component_sizes = ndimage.sum(mask_np, labeled, range(1, num_features + 1))
    sorted_indices = np.argsort(component_sizes)[::-1]
    out = np.zeros_like(mask_np)
    for i in range(min(num_keep, num_features)):
        comp_label = sorted_indices[i] + 1
        comp_size = component_sizes[sorted_indices[i]]
        if min_size > 0 and comp_size < min_size:
            break
        out[labeled == comp_label] = 1
    return out


def get_tumor_center(label_np):
    """Return center of mass (d, h, w) of the foreground in the label."""
    coords = np.argwhere(label_np > 0)
    if len(coords) == 0:
        return np.array(label_np.shape) // 2
    return coords.mean(axis=0).astype(int)


def get_crop_slices(center, roi_size, vol_shape):
    """Return (start, end) slices for a crop centered at `center`, clamped to bounds."""
    starts = []
    ends = []
    for i in range(3):
        half = roi_size[i] // 2
        s = center[i] - half
        s = max(0, min(s, vol_shape[i] - roi_size[i]))
        starts.append(s)
        ends.append(s + roi_size[i])
    return tuple(starts), tuple(ends)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"])
    parser.add_argument("--snapshot_path", default="", type=str)
    parser.add_argument("--data_prefix", default="", type=str)
    parser.add_argument("--rand_crop_size", default=0, nargs='+', type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument(
        "--checkpoint", default="last", type=str,
        help="Either 'last' / 'best' (loads <snapshot_path>/<name>.pth.tar) or an "
             "explicit path to a .pth.tar file.",
    )
    parser.add_argument("-tolerance", default=5, type=int)
    # DRLoRA-specific args
    parser.add_argument("--num_slices", default=32, type=int)
    parser.add_argument("--slice_size", default=512, type=int)
    parser.add_argument("--rank", default=16, type=int)
    parser.add_argument("--num_experts", default=4, type=int)
    parser.add_argument("--shift_ratio", default=0.25, type=float)
    parser.add_argument("--sam_checkpoint", default="ckpt/sam_vit_b_01ec64.pth", type=str)
    parser.add_argument("--decoder_type", default="conv3d", type=str, choices=["conv3d", "mla", "sam"])
    parser.add_argument("--save_pred", action="store_true", help="Save prediction masks")
    # Point-based crop settings
    parser.add_argument("--num_crops", default=5, type=int, help="Number of jittered crops per case")
    parser.add_argument("--jitter", default=15, type=int, help="Max random offset from tumor center (voxels)")
    parser.add_argument("--min_size", default=100, type=int, help="Min component size for post-processing")
    parser.add_argument("--num_points", default=1, type=int, help="Number of point prompts per crop (SAM decoder)")
    parser.add_argument("--seed", default=42, type=int)
    # ablation flags
    parser.add_argument("--no_z_embed", action="store_true", help="Ablation: disable z-position embedding")
    parser.add_argument("--no_decoder_lora", action="store_true", help="Ablation: disable decoder LoRA")
    parser.add_argument(
        "--routing_mode", default="depth", type=str,
        choices=["depth", "content", "content_top1", "content_top2", "single"],
        help="DRLoRA routing mode (must match the trained checkpoint)",
    )

    args = parser.parse_args()
    device = args.device

    if args.rand_crop_size == 0:
        if args.data in ["colon", "pancreas", "lits", "kits"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)

    args.snapshot_path = os.path.join(args.snapshot_path, args.data, args.decoder_type)
    setup_logger(logger_name="test_point", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger("test_point")
    logger.info(str(args))

    np.random.seed(args.seed)

    # Test data — full volume (no test crop) with dataset reference for original paths
    test_dataset = create_dataset(
        data=args.data,
        path_prefix=args.data_prefix,
        split="test",
        augmentation=False,
        rand_crop_spatial_size=args.rand_crop_size,
        convert_to_sam=False,
        do_test_crop=False,
    )
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Build model
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    model = SAM_D(
        sam_model=sam,
        num_slices=args.num_slices,
        slice_size=args.slice_size,
        rank=args.rank,
        num_experts=args.num_experts,
        shift_ratio=args.shift_ratio,
        num_classes=args.num_classes,
        decoder_type=args.decoder_type,
        no_z_embed=args.no_z_embed,
        no_decoder_lora=args.no_decoder_lora,
        routing_mode=args.routing_mode,
    )
    del sam

    # --checkpoint can be a shortcut ("last"/"best") or a full path to a .pth.tar
    if args.checkpoint in ("last", "best"):
        ckpt_path = os.path.join(args.snapshot_path, f"{args.checkpoint}.pth.tar")
    else:
        ckpt_path = args.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(migrate_legacy_state_dict(ckpt["state_dict"]))
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    if args.save_pred:
        pred_dir = os.path.join(args.snapshot_path, "predictions_point")
        os.makedirs(pred_dir, exist_ok=True)
        logger.info(f"Saving predictions to {pred_dir}")

    roi = args.rand_crop_size
    logger.info(
        f"ROI size: {roi}, num_crops: {args.num_crops}, "
        f"jitter: ±{args.jitter}, min_size: {args.min_size}"
    )

    dice_list = []
    nsd_list = []

    with torch.no_grad():
        for i, (image, label, spacing) in enumerate(test_data):
            # image: (1, 3, D, H, W), label: (1, D, H, W)
            vol_shape = tuple(image.shape[2:])
            label_np = label[0].numpy()  # (D, H, W)
            num_classes = args.num_classes

            # Pad volume if any dim < roi
            pad_offsets = [0, 0, 0]
            pad_needed = any(vol_shape[d] < roi[d] for d in range(3))
            if pad_needed:
                pad_size = [max(roi[d] - vol_shape[d], 0) for d in range(3)]
                pad_arg = []
                for d in reversed(range(3)):
                    pad_arg.extend([pad_size[d] // 2, pad_size[d] - pad_size[d] // 2])
                image = F.pad(image, pad_arg)
                label = F.pad(label, pad_arg)
                pad_offsets = [pad_size[d] // 2 for d in range(3)]
                vol_shape = tuple(image.shape[2:])
                label_np = label[0].numpy()

            # Accumulate softmax probabilities over the full volume
            prob_sum = torch.zeros(num_classes, *vol_shape, device=device)
            count_map = torch.zeros(*vol_shape, device=device)

            center = get_tumor_center(label_np)

            crop_centers = []
            for c in range(args.num_crops):
                jitter = np.random.randint(-args.jitter, args.jitter + 1, size=3)
                cj = np.clip(center + jitter, 0, np.array(vol_shape) - 1)
                crop_centers.append(cj)

            for c, cj in enumerate(crop_centers):
                starts, ends = get_crop_slices(cj, roi, vol_shape)
                sd, sh, sw = starts
                ed, eh, ew = ends

                img_crop = image[0, :, sd:ed, sh:eh, sw:ew].unsqueeze(0).to(device)

                with torch.amp.autocast("cuda"):
                    if model.decoder_type == "sam":
                        # Fixed point prompt: tumor center in crop-relative coords
                        # Same absolute point for all crops, only crop position changes
                        crop_z = float(center[0] - sd)
                        crop_h = float(center[1] - sh)
                        crop_w = float(center[2] - sw)
                        pt_coords = torch.tensor([[[crop_z, crop_h, crop_w]]], device=device)
                        pt_labels = torch.ones(1, 1, dtype=torch.long, device=device)
                        logits = model(img_crop, point_coords=pt_coords, point_labels=pt_labels)
                    else:
                        logits = model(img_crop)  # (1, C, rd, rh, rw)

                probs = F.softmax(logits[0], dim=0)  # (C, rd, rh, rw)
                prob_sum[:, sd:ed, sh:eh, sw:ew] += probs
                count_map[sd:ed, sh:eh, sw:ew] += 1

            # Average probabilities where we have predictions
            mask_covered = count_map > 0
            for cls in range(num_classes):
                prob_sum[cls][mask_covered] /= count_map[mask_covered]

            # Argmax -> prediction
            pred_vol = prob_sum.argmax(dim=0)  # (D, H, W)

            # Post-processing: keep largest component, remove noise
            pred_np = pred_vol.cpu().numpy().astype(np.uint8)
            pred_np = keep_largest_components(pred_np, num_keep=1, min_size=args.min_size)
            pred_vol = torch.from_numpy(pred_np).to(device)

            # Metrics on full volume
            label_vol = label[0].to(device)
            dice = compute_dice(pred_vol == 1, label_vol == 1)
            dice_list.append(dice)

            spacing_np = spacing[0].numpy() if hasattr(spacing[0], 'numpy') else np.array(spacing[0])
            nsd = compute_nsd(pred_vol == 1, label_vol == 1, spacing_np, tolerance=args.tolerance)
            nsd_list.append(nsd)

            pred_fg = (pred_vol == 1).sum().item()
            label_fg = (label_vol == 1).sum().item()
            n_covered = mask_covered.sum().item()
            n_total = np.prod(vol_shape)
            logger.info(
                f"Case [{i+1}/{len(test_data)}] "
                f"Dice: {dice:.4f} NSD: {nsd:.4f} "
                f"pred_fg: {pred_fg} label_fg: {label_fg} "
                f"covered: {n_covered}/{n_total} ({100*n_covered/n_total:.1f}%) "
                f"vol: {vol_shape} center: {center.tolist()}"
            )

            if args.save_pred:
                # Resample prediction back to original NIfTI size
                orig_nii = nib.load(test_dataset.img_dict[i])
                orig_affine = orig_nii.affine
                orig_shape = np.array(orig_nii.shape)[test_dataset.spatial_index]
                # pred_np is in resampled (target_spacing) space, shape = vol_shape
                # Resize to original shape using nearest interpolation
                pred_tensor = torch.from_numpy(pred_np).float().unsqueeze(0).unsqueeze(0)
                pred_resized = F.interpolate(
                    pred_tensor, size=tuple(orig_shape), mode='nearest'
                ).squeeze().numpy().astype(np.uint8)
                # Undo spatial_index reorder to match original axis order
                inverse_index = np.argsort(test_dataset.spatial_index)
                pred_original = pred_resized.transpose(inverse_index)
                nii = nib.Nifti1Image(pred_original, orig_affine)
                # Use original case name for the saved file
                img_path = test_dataset.img_dict[i]
                case_name = os.path.splitext(os.path.splitext(os.path.basename(img_path))[0])[0]
                if case_name in ("image", "imaging"):
                    case_name = os.path.basename(os.path.dirname(img_path))
                nib.save(nii, os.path.join(pred_dir, f"{case_name}.nii.gz"))

    mean_dice = np.mean(dice_list)
    mean_nsd = np.mean(nsd_list)
    logger.info(f"Mean Dice: {mean_dice:.4f} | Mean NSD: {mean_nsd:.4f}")
    logger.info(f"Std  Dice: {np.std(dice_list):.4f} | Std  NSD: {np.std(nsd_list):.4f}")


if __name__ == "__main__":
    main()
