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
from monai.inferers import sliding_window_inference


def compute_dice(pred, label, num_classes=2):
    """Compute per-class Dice score (excluding background class 0)."""
    dice_scores = []
    for c in range(1, num_classes):
        pred_c = (pred == c).float()
        label_c = (label == c).float()
        intersection = (pred_c * label_c).sum()
        union = pred_c.sum() + label_c.sum()
        if union == 0:
            dice_scores.append(1.0)
        else:
            dice_scores.append((2.0 * intersection / union).item())
    return np.mean(dice_scores)


def compute_nsd(pred, label, spacing, tolerance=5):
    """Compute Normalized Surface Distance for foreground class."""
    pred_np = pred.cpu().numpy().astype(bool)
    label_np = label.cpu().numpy().astype(bool)
    if not pred_np.any() and not label_np.any():
        return 1.0
    if not pred_np.any() or not label_np.any():
        return 0.0
    surface_distances = surface_distance.compute_surface_distances(
        label_np, pred_np, spacing_mm=spacing
    )
    nsd = metrics.compute_surface_dice_at_tolerance(surface_distances, tolerance)
    return nsd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"]
    )
    parser.add_argument("--snapshot_path", default="", type=str)
    parser.add_argument("--data_prefix", default="", type=str)
    parser.add_argument(
        "--rand_crop_size", default=0, nargs='+', type=int,
    )
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
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
    parser.add_argument("--save_pred", action="store_true", help="Save prediction masks as NIfTI files")
    parser.add_argument("--num_points", default=1, type=int, help="Number of point prompts per window (SAM decoder)")
    # Sliding window inference
    parser.add_argument("--sw_batch_size", default=4, type=int, help="Sliding window batch size")
    parser.add_argument("--overlap", default=0.5, type=float, help="Sliding window overlap ratio")

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

    setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger("test")
    logger.info(str(args))

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
    )
    del sam

    # Load trained weights: --checkpoint can be a shortcut ("last"/"best") or a path
    if args.checkpoint in ("last", "best"):
        ckpt_path = os.path.join(args.snapshot_path, f"{args.checkpoint}.pth.tar")
    else:
        ckpt_path = args.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(migrate_legacy_state_dict(ckpt["state_dict"]))
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

    # Prediction save directory
    if args.save_pred:
        pred_dir = os.path.join(args.snapshot_path, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        logger.info(f"Saving predictions to {pred_dir}")

    logger.info(
        f"Sliding window inference: roi_size={args.rand_crop_size}, "
        f"sw_batch_size={args.sw_batch_size}, overlap={args.overlap}"
    )

    # Evaluate
    dice_list = []
    nsd_list = []

    with torch.no_grad():
        for i, (image, label, spacing) in enumerate(test_data):
            # image: (1, 3, D, H, W), label: (1, D, H, W)
            image = image.to(device)
            label = label.to(device)

            # Sliding window inference: process overlapping windows of rand_crop_size
            with torch.amp.autocast("cuda"):
                if model.decoder_type == "sam":
                    n_pts = args.num_points
                    # For SAM decoder, wrap predictor to inject point prompts
                    def sam_predictor(window, _n_pts=n_pts):
                        # window: (sw_batch, 3, D, H, W)
                        B_w = window.shape[0]
                        D_w, H_w, W_w = window.shape[2:]
                        # Sample random points within the center region of the window
                        coords = []
                        for _ in range(_n_pts):
                            d = D_w / 2.0 + (torch.rand(1).item() - 0.5) * D_w * 0.3
                            h = H_w / 2.0 + (torch.rand(1).item() - 0.5) * H_w * 0.3
                            w = W_w / 2.0 + (torch.rand(1).item() - 0.5) * W_w * 0.3
                            coords.append([d, h, w])
                        pts = torch.tensor([coords], device=window.device).expand(B_w, -1, -1)
                        labels = torch.ones(B_w, _n_pts, dtype=torch.long, device=window.device)
                        return model(window, point_coords=pts, point_labels=labels)
                    logits = sliding_window_inference(
                        image,
                        roi_size=args.rand_crop_size,
                        sw_batch_size=args.sw_batch_size,
                        predictor=sam_predictor,
                        overlap=args.overlap,
                        mode="gaussian",
                    )
                else:
                    logits = sliding_window_inference(
                        image,
                        roi_size=args.rand_crop_size,
                        sw_batch_size=args.sw_batch_size,
                        predictor=model,
                        overlap=args.overlap,
                        mode="gaussian",
                    )

            # Safety: resize logits to match label size if shapes differ
            if logits.shape[2:] != label.shape[1:]:
                logits = F.interpolate(logits, size=label.shape[1:], mode='trilinear', align_corners=False)

            pred = logits.argmax(dim=1)  # (1, D, H, W)

            # Save prediction mask as NIfTI — resample to original size
            if args.save_pred:
                pred_np = pred[0].cpu().numpy().astype(np.uint8)  # (D, H, W)
                orig_nii = nib.load(test_dataset.img_dict[i])
                orig_affine = orig_nii.affine
                orig_shape = np.array(orig_nii.shape)[test_dataset.spatial_index]
                pred_tensor = torch.from_numpy(pred_np).float().unsqueeze(0).unsqueeze(0)
                pred_resized = F.interpolate(
                    pred_tensor, size=tuple(orig_shape), mode='nearest'
                ).squeeze().numpy().astype(np.uint8)
                inverse_index = np.argsort(test_dataset.spatial_index)
                pred_original = pred_resized.transpose(inverse_index)
                nii = nib.Nifti1Image(pred_original, orig_affine)
                # Use original case name for the saved file
                img_path = test_dataset.img_dict[i]
                case_name = os.path.splitext(os.path.splitext(os.path.basename(img_path))[0])[0]
                if case_name in ("image", "imaging"):
                    case_name = os.path.basename(os.path.dirname(img_path))
                nib.save(nii, os.path.join(pred_dir, f"{case_name}.nii.gz"))

            # Dice
            dice = compute_dice(pred, label, args.num_classes)
            dice_list.append(dice)

            # NSD
            spacing_np = spacing[0].numpy() if hasattr(spacing[0], 'numpy') else np.array(spacing[0])
            nsd = compute_nsd(
                pred[0] == 1, label[0] == 1, spacing_np, tolerance=args.tolerance
            )
            nsd_list.append(nsd)

            pred_fg = (pred == 1).sum().item()
            label_fg = (label == 1).sum().item()
            logger.info(
                f"Case [{i+1}/{len(test_data)}] "
                f"Dice: {dice:.4f} NSD: {nsd:.4f} "
                f"pred_fg: {pred_fg} label_fg: {label_fg} "
                f"vol_shape: {tuple(image.shape[2:])}"
            )

    mean_dice = np.mean(dice_list)
    mean_nsd = np.mean(nsd_list)
    logger.info(f"Mean Dice: {mean_dice:.4f} | Mean NSD: {mean_nsd:.4f}")
    logger.info(f"Std Dice: {np.std(dice_list):.4f} | Std NSD: {np.std(nsd_list):.4f}")


if __name__ == "__main__":
    main()
