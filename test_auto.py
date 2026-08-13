from dataset.datasets import load_data_volume
from segment_anything import sam_model_registry
import argparse
import numpy as np
import logging
from monai.losses import DiceCELoss, DiceLoss
import torch.nn.functional as F
import torch
from functools import partial
import os
from utils.util import setup_logger
import surface_distance
from surface_distance import metrics
from monai.inferers import sliding_window_inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon", "nc_liver"]
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument(
        "--checkpoint",
        default="last",
        type=str,
    )
    parser.add_argument("-tolerance", default=5, type=int)
    args = parser.parse_args()
    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"
    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["colon", "pancreas", "lits", "kits", "nc_liver"]:
            args.rand_crop_size = (256, 256, 256)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)

    setup_logger(logger_name="test", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"test")
    logger.info(str(args))
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
        num_worker=0
    )

    # Load pure SAM1 model
    sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")
    img_encoder = sam.image_encoder
    img_encoder.to(device)

    # TODO: Add your own 3D adaptation method and load trained weights here
    # The SAM1 image encoder is a 2D ViT — it cannot directly process 3D volumes.

    raise NotImplementedError(
        "Pure SAM1 encoder loaded. Add your own 3D adaptation and testing logic."
    )


if __name__ == "__main__":
    main()
