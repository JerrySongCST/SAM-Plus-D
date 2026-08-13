"""Post-processing: keep only the largest connected component(s) in prediction masks."""
import argparse
import os
import numpy as np
import nibabel as nib
from scipy import ndimage


def keep_largest_components(mask, num_keep=1, min_size=0):
    """Keep the largest connected component(s) in a binary mask.

    Args:
        mask: binary numpy array (D, H, W)
        num_keep: number of largest components to keep
        min_size: minimum voxel count to keep a component (0 = disabled)
    Returns:
        cleaned binary mask
    """
    if not mask.any():
        return mask

    labeled, num_features = ndimage.label(mask)
    if num_features <= num_keep and min_size == 0:
        return mask

    # Get component sizes (skip background label 0)
    component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    # Sort by size descending
    sorted_indices = np.argsort(component_sizes)[::-1]

    out = np.zeros_like(mask)
    for i in range(min(num_keep, num_features)):
        comp_label = sorted_indices[i] + 1  # +1 because labels start at 1
        comp_size = component_sizes[sorted_indices[i]]
        if min_size > 0 and comp_size < min_size:
            break
        out[labeled == comp_label] = 1

    return out


def main():
    parser = argparse.ArgumentParser(description="Post-process prediction masks")
    parser.add_argument("--input_dir", required=True, type=str, help="Input predictions directory")
    parser.add_argument("--output_dir", default=None, type=str, help="Output directory (default: input_dir + '_postprocessed')")
    parser.add_argument("--num_keep", default=1, type=int, help="Number of largest components to keep")
    parser.add_argument("--min_size", default=100, type=int, help="Minimum component size in voxels (smaller are removed)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir.rstrip("/") + "_postprocessed"
    os.makedirs(args.output_dir, exist_ok=True)

    nii_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith((".nii.gz", ".nii"))])
    print(f"Found {len(nii_files)} files in {args.input_dir}")
    print(f"Settings: num_keep={args.num_keep}, min_size={args.min_size}")
    print(f"Output: {args.output_dir}")

    for fname in nii_files:
        nii = nib.load(os.path.join(args.input_dir, fname))
        mask = nii.get_fdata().astype(np.uint8)

        fg_before = mask.sum()
        cleaned = keep_largest_components(mask, num_keep=args.num_keep, min_size=args.min_size)
        fg_after = int(cleaned.sum())

        # Count components before and after
        _, n_before = ndimage.label(mask)
        _, n_after = ndimage.label(cleaned)

        out_nii = nib.Nifti1Image(cleaned.astype(np.uint8), nii.affine, nii.header)
        nib.save(out_nii, os.path.join(args.output_dir, fname))

        removed = fg_before - fg_after
        print(f"  {fname}: {n_before} components -> {n_after}, removed {removed} voxels ({100*removed/max(fg_before,1):.1f}%)")

    print("Done.")


if __name__ == "__main__":
    main()
