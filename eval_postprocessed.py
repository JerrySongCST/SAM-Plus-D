"""Evaluate post-processed predictions against ground truth labels."""
import argparse
import os
import numpy as np
import nibabel as nib
import torch
import surface_distance
from surface_distance import metrics
from dataset.datasets import load_data_volume


def compute_dice(pred, label):
    """Compute Dice score for binary masks."""
    pred = pred.astype(bool)
    label = label.astype(bool)
    if not pred.any() and not label.any():
        return 1.0
    if not pred.any() or not label.any():
        return 0.0
    intersection = (pred & label).sum()
    return 2.0 * intersection / (pred.sum() + label.sum())


def compute_nsd(pred, label, spacing, tolerance=5):
    """Compute Normalized Surface Distance."""
    pred = pred.astype(bool)
    label = label.astype(bool)
    if not pred.any() and not label.any():
        return 1.0
    if not pred.any() or not label.any():
        return 0.0
    surface_distances = surface_distance.compute_surface_distances(
        label, pred, spacing_mm=spacing
    )
    return metrics.compute_surface_dice_at_tolerance(surface_distances, tolerance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, type=str)
    parser.add_argument("--data", default="kits", type=str, choices=["kits", "pancreas", "lits", "colon"])
    parser.add_argument("--data_prefix", default="", type=str)
    parser.add_argument("--rand_crop_size", default=(128, 128, 128), nargs='+', type=int)
    parser.add_argument("-tolerance", default=5, type=int)
    args = parser.parse_args()

    if len(args.rand_crop_size) == 1:
        args.rand_crop_size = tuple(args.rand_crop_size * 3)
    else:
        args.rand_crop_size = tuple(args.rand_crop_size)

    # Load test dataset (same pipeline as test.py)
    test_data = load_data_volume(
        data=args.data,
        batch_size=1,
        path_prefix=args.data_prefix,
        augmentation=False,
        split="test",
        rand_crop_spatial_size=args.rand_crop_size,
        convert_to_sam=False,
        do_test_crop=False,
        deterministic=True,
        num_worker=0,
    )

    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith((".nii.gz", ".nii"))])
    print(f"Found {len(pred_files)} predictions in {args.pred_dir}")
    print(f"Test dataset has {len(test_data)} cases")

    dice_list = []
    nsd_list = []

    for i, (image, label, spacing) in enumerate(test_data):
        pred_path = os.path.join(args.pred_dir, f"case_{i:04d}.nii.gz")
        if not os.path.exists(pred_path):
            print(f"  WARNING: {pred_path} not found, skipping")
            continue

        pred_nii = nib.load(pred_path)
        pred_np = pred_nii.get_fdata().astype(np.uint8)

        # label: (1, D, H, W) -> (D, H, W)
        label_np = label[0].numpy().astype(np.uint8)

        # Check shape match
        if pred_np.shape != label_np.shape:
            print(f"  WARNING case {i}: pred shape {pred_np.shape} != label shape {label_np.shape}, resizing pred")
            pred_tensor = torch.tensor(pred_np[None, None].astype(np.float32))
            pred_tensor = torch.nn.functional.interpolate(
                pred_tensor, size=label_np.shape, mode='nearest'
            )
            pred_np = pred_tensor[0, 0].numpy().astype(np.uint8)

        spacing_np = spacing[0].numpy() if hasattr(spacing[0], 'numpy') else np.array(spacing[0])

        dice = compute_dice(pred_np, label_np)
        nsd = compute_nsd(pred_np, label_np, spacing_np, tolerance=args.tolerance)

        dice_list.append(dice)
        nsd_list.append(nsd)

        pred_fg = pred_np.sum()
        label_fg = label_np.sum()
        print(f"  Case [{i+1}/{len(test_data)}] Dice: {dice:.4f} NSD: {nsd:.4f} pred_fg: {pred_fg} label_fg: {label_fg}")

    print(f"\nMean Dice: {np.mean(dice_list):.4f} | Mean NSD: {np.mean(nsd_list):.4f}")
    print(f"Std  Dice: {np.std(dice_list):.4f} | Std  NSD: {np.std(nsd_list):.4f}")


if __name__ == "__main__":
    main()
