# SAM+D

**Official PyTorch implementation of**

> **SAM+D: Parameter-Efficient Dimensional Lifting of SAM-Family Models via Depth-Routed LoRA and Depth Shifting**
> *European Conference on Computer Vision (ECCV) 2026*

---

## Abstract

The Segment Anything Model family (SAM, SAM2) provides powerful 2D and 2D+T segmentation foundation models, but extending them to volumetric domains such as 3D medical imaging remains expensive and ad-hoc — typically demanding full fine-tuning or heavy 3D-specific adapters. We propose **SAM+D**, a unified parameter-efficient framework that lifts SAM-family backbones to 3D / 3D+T by injecting two lightweight modules into the frozen 2D image encoder:

- **DRLoRA** (Depth-Routed Mixture of LoRA Experts) — a per-slice mixture of low-rank adapters whose gates are conditioned on the slice position *z* alone. DRLoRA supplies an anatomical depth prior that the frozen backbone cannot recover, is inherently load-balanced (no auxiliary loss needed), and uses a router roughly twenty times smaller than content-routed mixtures of LoRA experts.
- **DSM** (Depth Shift Module) — a parameter-free, zero-MAC channel-shift primitive along the depth axis that propagates inter-slice information at a cost of roughly 1.7 % of a single ViT-B block's forward latency on an NVIDIA RTX 5090.

The same module pair lifts both **SAM (2D → 3D)** and **SAM2 (2D+T → 3D+T)** without per-task redesign. On four public 3D medical segmentation benchmarks (KiTS, LiTS, Pancreas, Colon) and a private non-contrast CT liver-tumor cohort, SAM+D matches or exceeds 3D-conv-adapter baselines and recent mixture-of-LoRA approaches (MoLoRA, MoLE, MixLoRA), with **fewer than 3 % of the backbone parameters trained**.

---

## Code Release

**The code, pretrained checkpoints, configuration files, and reproducibility scripts will be released here soon.** Please watch / star this repository to be notified.

Planned contents:
- Training and evaluation scripts for SAM and SAM2 backbones
- DRLoRA and DSM module implementations
- Configurations for all reported datasets
- Pretrained checkpoints for the main results table
- Ablation scripts (routing-mechanism, shift ratio, experts, rank)

---
```

---

## Contact

For questions, collaboration inquiries, or early-access requests, please open an issue on this repository.
