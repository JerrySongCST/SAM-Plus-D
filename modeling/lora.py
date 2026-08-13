import torch
import torch.nn as nn
import math


class LoRAExpert(nn.Module):
    """Single low-rank adaptation: x -> down(in_dim -> rank) -> up(rank -> in_dim)."""

    def __init__(self, in_dim: int = 768, rank: int = 16):
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, in_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank correction.

    Used for PEFT on SAM's mask decoder attention layers (q_proj, v_proj).
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 16):
        super().__init__()
        self.base = base_linear
        self.lora_down = nn.Linear(base_linear.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x))


VALID_ROUTING_MODES = ("depth", "content", "content_top1", "content_top2", "single")


class DRLoRA(nn.Module):
    """Mixture of LoRA experts with pluggable routing.

    routing_mode:
      - "depth"          (DRLoRA, ours): router on slice z in R^1, dense softmax, no aux loss.
      - "content"        (DRLoRA baseline): router on mean-pooled features in R^C, dense softmax.
      - "content_top1"   (MixLoRA): top-1 switch on content + Switch load-balance aux loss.
      - "content_top2"   (MoLE):   top-2 on content + importance (CV^2) aux loss.
      - "single"         (or num_experts==1): bypass router, single LoRA expert.

    forward returns (out, aux_loss). aux_loss is a scalar tensor (0 when not used).
    """

    def __init__(
        self,
        in_dim: int = 768,
        rank: int = 16,
        num_experts: int = 4,
        routing_mode: str = "depth",
    ):
        super().__init__()
        if routing_mode not in VALID_ROUTING_MODES:
            raise ValueError(f"routing_mode must be one of {VALID_ROUTING_MODES}, got {routing_mode!r}")
        self.experts = nn.ModuleList(
            [LoRAExpert(in_dim, rank) for _ in range(num_experts)]
        )
        self.num_experts = num_experts
        self.in_dim = in_dim
        self.routing_mode = "single" if num_experts == 1 else routing_mode

        if self.routing_mode == "single":
            self.router = None
        elif self.routing_mode == "depth":
            # ~140 params for E=4: position-only router (DRLoRA, ours)
            self.router = nn.Sequential(
                nn.Linear(1, num_experts * 16),
                nn.GELU(),
                nn.Linear(num_experts * 16, num_experts),
            )
        else:
            # content-based routers all share the same architecture: linear C -> E
            self.router = nn.Linear(in_dim, num_experts)

    def forward(self, x: torch.Tensor, slice_position: torch.Tensor):
        """
        Args:
            x: (B, H, W, C)
            slice_position: (B, 1) — used only by depth routing
        Returns:
            (out, aux_loss)
            out: (B, H, W, C)
            aux_loss: scalar tensor
        """
        zero_aux = x.new_zeros(())

        if self.routing_mode == "single":
            return self.experts[0](x), zero_aux

        if self.routing_mode == "depth":
            logits = self.router(slice_position)             # (B, E)
            weights = logits.softmax(dim=-1)
            out = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                w = weights[:, i].view(-1, 1, 1, 1)
                out = out + w * expert(x)
            return out, zero_aux

        # Content-based routing: pool spatial dims to (B, C)
        content_feat = x.mean(dim=(1, 2))                    # (B, C)
        logits = self.router(content_feat)                   # (B, E)
        weights = logits.softmax(dim=-1)                     # (B, E)

        if self.routing_mode == "content":
            # Dense softmax over all experts
            out = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                w = weights[:, i].view(-1, 1, 1, 1)
                out = out + w * expert(x)
            return out, zero_aux

        if self.routing_mode == "content_top1":
            # Switch-style top-1 routing + load-balance aux loss (eq. 5/6 from Switch Transformer).
            # We keep the *compute* dense (E experts evaluated) for fair per-step comparison;
            # the routing structure (one expert per token-batch) is what we ablate.
            top1_idx = logits.argmax(dim=-1)                                # (B,)
            sparse_w = torch.zeros_like(weights)
            sparse_w.scatter_(1, top1_idx.unsqueeze(-1),
                              weights.gather(1, top1_idx.unsqueeze(-1)))    # (B, E) one nonzero per row
            out = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                w = sparse_w[:, i].view(-1, 1, 1, 1)
                out = out + w * expert(x)
            # f_i = fraction routed to expert i (hard); P_i = mean soft prob to expert i
            f = torch.zeros(self.num_experts, device=x.device, dtype=weights.dtype)
            f = f.scatter_add(0, top1_idx, torch.ones_like(top1_idx, dtype=weights.dtype)) / max(1, top1_idx.numel())
            P = weights.mean(dim=0)
            aux_loss = self.num_experts * (f * P).sum()
            return out, aux_loss

        if self.routing_mode == "content_top2":
            # MoLE-style top-2 dense + importance (CV^2) aux loss.
            k = min(2, self.num_experts)
            topk_vals, topk_idx = weights.topk(k, dim=-1)                   # (B, k), (B, k)
            topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)
            sparse_w = torch.zeros_like(weights).scatter_(1, topk_idx, topk_vals)
            out = torch.zeros_like(x)
            for i, expert in enumerate(self.experts):
                w = sparse_w[:, i].view(-1, 1, 1, 1)
                out = out + w * expert(x)
            importance = weights.sum(dim=0)                                  # (E,)
            cv2 = (importance.std() / (importance.mean() + 1e-9)) ** 2
            return out, cv2

        raise RuntimeError(f"unreachable routing_mode: {self.routing_mode}")
