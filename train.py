from dataset.datasets import create_dataset, load_data_volume
from segment_anything import sam_model_registry
from modeling.sam_d import SAM_D
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
from utils.util import migrate_legacy_state_dict
import sys
from monai.losses import DiceCELoss
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
import os
from utils.util import setup_logger
import wandb


def sample_foreground_point(label, num_points=1):
    """Sample random foreground (z, h, w) points from each batch element.

    Args:
        label: (B, D, H, W) — ground truth segmentation
        num_points: number of points to sample per batch element
    Returns:
        point_coords: (B, num_points, 3) — (z, h, w) coordinates
        point_labels: (B, num_points) — 1=positive, 0=negative
    """
    B = label.shape[0]
    device = label.device
    coords_list, labels_list = [], []
    for b in range(B):
        fg = (label[b] > 0).nonzero(as_tuple=False)  # (N, 3) = (d, h, w)
        if len(fg) > 0:
            idx = torch.randint(len(fg), (num_points,), device=device)
            coords_list.append(fg[idx])  # (num_points, 3)
            labels_list.append(torch.ones(num_points, dtype=torch.long, device=device))
        else:
            D, H, W = label.shape[1:]
            center = torch.tensor([[D // 2, H // 2, W // 2]], device=device)
            coords_list.append(center.expand(num_points, -1))
            labels_list.append(torch.zeros(num_points, dtype=torch.long, device=device))
    return torch.stack(coords_list), torch.stack(labels_list)


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


def setup_distributed():
    """Initialize DDP from torchrun environment variables.
    Returns (rank, local_rank, world_size) or (0, 0, 1) for single-GPU.
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


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
    parser.add_argument("-bs", "--batch_size", default=3, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_worker", default=6, type=int)
    parser.add_argument("-tolerance", default=5, type=int)
    # DRLoRA-specific args
    parser.add_argument("--num_slices", default=32, type=int)
    parser.add_argument("--slice_size", default=512, type=int)
    parser.add_argument("--rank", default=16, type=int)
    parser.add_argument("--num_experts", default=4, type=int)
    parser.add_argument("--shift_ratio", default=0.25, type=float)
    parser.add_argument("--sam_checkpoint", default="ckpt/sam_vit_b_01ec64.pth", type=str)
    parser.add_argument("--decoder_type", default="conv3d", type=str, choices=["conv3d", "mla", "sam"])
    parser.add_argument("--num_points", default=1, type=int, help="Number of point prompts per sample (SAM decoder)")
    # wandb args
    parser.add_argument("--wandb_project", default="sam-plus-d", type=str)
    parser.add_argument("--wandb_run", default=None, type=str)
    parser.add_argument("--no_wandb", action="store_true")
    # ablation flags
    parser.add_argument("--no_z_embed", action="store_true", help="Ablation: disable z-position embedding")
    parser.add_argument("--no_decoder_lora", action="store_true", help="Ablation: disable decoder LoRA")
    # routing-mechanism ablation
    parser.add_argument(
        "--routing_mode", default="depth", type=str,
        choices=["depth", "content", "content_top1", "content_top2", "single"],
        help="Routing mode: depth=DRLoRA(ours), content=MoLoRA, content_top1=MixLoRA, content_top2=MoLE, single=baseline LoRA",
    )
    parser.add_argument("--aux_loss_weight", default=0.01, type=float, help="Weight for routing aux loss (top-k modes)")
    # gradient accumulation & clipping
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)

    args = parser.parse_args()

    # DDP setup
    rank, local_rank, world_size = setup_distributed()
    distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    if args.rand_crop_size == 0:
        if args.data in ["pancreas", "lits", "colon", "kits"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    args.snapshot_path = os.path.join(args.snapshot_path, args.data, args.decoder_type)
    if is_main and not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    if distributed:
        dist.barrier()

    # Logger
    if is_main:
        setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    else:
        setup_logger(logger_name="train", root=args.snapshot_path, screen=False, tofile=False)
    logger = logging.getLogger("train")
    logger.info(f"[Rank {rank}] {args}")
    if distributed:
        logger.info(f"[Rank {rank}] Distributed training: {world_size} GPUs")

    # wandb init (rank 0 only)
    use_wandb = is_main and not args.no_wandb
    if use_wandb:
        run_name = args.wandb_run or f"{args.data}_{args.decoder_type}_r{args.rank}_e{args.num_experts}_s{args.num_slices}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            resume="allow" if args.resume else None,
        )

    # Data loading
    train_dataset = create_dataset(
        data=args.data,
        path_prefix=args.data_prefix,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_data = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_worker,
        drop_last=True,
    )

    val_data = load_data_volume(
        data=args.data,
        path_prefix=args.data_prefix,
        batch_size=1,
        augmentation=False,
        split="val",
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker=args.num_worker,
    )

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

    # Resume (load into unwrapped model before DDP)
    start_epoch = 0
    best_dice = 0.0
    resume_ckpt = None
    if args.resume:
        ckpt_path = os.path.join(args.snapshot_path, "last.pth.tar")
        if os.path.isfile(ckpt_path):
            resume_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(migrate_legacy_state_dict(resume_ckpt["state_dict"]))
            start_epoch = resume_ckpt["epoch"]
            best_dice = resume_ckpt.get("best_dice", 0.0)
            logger.info(f"Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")
        else:
            if is_main:
                logger.warning(f"No checkpoint found at {ckpt_path}, starting from scratch")

    model.to(device)

    # Log parameter counts (before DDP wrap)
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Wrap with DDP
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    model_unwrapped = model.module if distributed else model

    # Optimizer — only trainable parameters
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if resume_ckpt is not None and "optimizer" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        del resume_ckpt

    # Loss
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda")

    # Global step counter for wandb
    accum = args.grad_accum_steps
    global_step = start_epoch * ((len(train_data) + accum - 1) // accum)

    if is_main:
        eff_bs = args.batch_size * world_size * accum
        logger.info(f"Effective batch size: {args.batch_size} x {world_size} GPUs x {accum} accum = {eff_bs}")
        logger.info(f"Gradient clipping max_norm: {args.max_grad_norm}")

    # Training loop
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_loss = []

        optimizer.zero_grad()
        for i, (image, label, spacing) in enumerate(train_data):
            image = image.to(device)
            label = label.to(device)

            with torch.amp.autocast("cuda"):
                if model_unwrapped.decoder_type == "sam":
                    point_coords, point_labels = sample_foreground_point(label, args.num_points)
                    logits, aux_loss = model(image, point_coords=point_coords, point_labels=point_labels)
                else:
                    logits, aux_loss = model(image)
                loss = loss_fn(logits, label.unsqueeze(1).long())
                if args.aux_loss_weight > 0:
                    loss = loss + args.aux_loss_weight * aux_loss
                loss = loss / accum  # scale loss for accumulation

            scaler.scale(loss).backward()

            loss_val = loss.item() * accum  # unscaled for logging
            epoch_loss.append(loss_val)

            if (i + 1) % accum == 0 or (i + 1) == len(train_data):
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    args.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            if is_main and (i + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{args.max_epoch}] "
                    f"Step [{i+1}/{len(train_data)}] "
                    f"Loss: {loss_val:.4f}"
                )
            if use_wandb and ((i + 1) % accum == 0 or (i + 1) == len(train_data)):
                wandb.log({"train/loss_step": loss_val, "train/step": global_step}, step=global_step)

        mean_loss = np.mean(epoch_loss)
        if is_main:
            logger.info(f"Epoch [{epoch+1}/{args.max_epoch}] Mean Loss: {mean_loss:.4f}")
        if use_wandb:
            wandb.log({
                "train/loss_epoch": mean_loss,
                "train/epoch": epoch + 1,
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

        # Save last checkpoint every epoch (rank 0 only)
        if is_main:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_unwrapped.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_dice": best_dice,
                },
                is_best=False,
                checkpoint=args.snapshot_path,
            )

        # Validation (all ranks run forward to stay in sync, only rank 0 logs/saves)
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            dice_scores = []

            with torch.no_grad():
                for image, label, spacing in val_data:
                    image = image.to(device)
                    label = label.to(device)

                    with torch.amp.autocast("cuda"):
                        if model_unwrapped.decoder_type == "sam":
                            point_coords, point_labels = sample_foreground_point(label, args.num_points)
                            logits = model(image, point_coords=point_coords, point_labels=point_labels)
                        else:
                            logits = model(image)
                    pred = logits.argmax(dim=1)
                    dice = compute_dice(pred, label, args.num_classes)
                    dice_scores.append(dice)

            if is_main:
                mean_dice = np.mean(dice_scores)
                is_best = mean_dice > best_dice
                if is_best:
                    best_dice = mean_dice
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "state_dict": model_unwrapped.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_dice": best_dice,
                        },
                        is_best=True,
                        checkpoint=args.snapshot_path,
                    )

                logger.info(
                    f"Validation Epoch [{epoch+1}] "
                    f"Dice: {mean_dice:.4f} | Best: {best_dice:.4f}"
                )
                if use_wandb:
                    wandb.log({
                        "val/dice": mean_dice,
                        "val/best_dice": best_dice,
                        "val/epoch": epoch + 1,
                    }, step=global_step)

        # Sync all ranks before next epoch
        if distributed:
            dist.barrier()

    if use_wandb:
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
