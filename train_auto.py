from dataset.datasets import load_data_volume
from segment_anything import sam_model_registry
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
import sys
from monai.losses import DiceCELoss, DiceLoss
import torch.nn.functional as F
import torch
import torch.nn as nn
from functools import partial
import os
from utils.util import setup_logger

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
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("-tolerance", default=5, type=int)

    args = parser.parse_args()
    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["pancreas", "lits", "colon", "kits", "nc_liver"]:
            args.rand_crop_size = (256, 256, 256)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))
    train_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=1,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker
    )
    val_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=1,
        augmentation=False,
        split="val",
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker = args.num_worker
    )

    # Load pure SAM1 model
    sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")
    img_encoder = sam.image_encoder
    img_encoder.to(device)

    # TODO: Add your own 3D adaptation method here
    # The SAM1 image encoder is a 2D ViT — it cannot directly process 3D volumes.
    # You need to implement your own method to bridge 2D SAM to 3D input.

    raise NotImplementedError(
        "Pure SAM1 encoder loaded. Add your own 3D adaptation method before training."
    )


if __name__ == "__main__":
    main()
