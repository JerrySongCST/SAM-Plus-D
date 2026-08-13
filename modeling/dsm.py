import torch


def depth_shift(x: torch.Tensor, shift_ratio: float = 0.25) -> torch.Tensor:
    """Zero-parameter Temporal Shift Module for cross-slice information flow.

    Shifts a fraction of channels forward/backward along the slice (batch)
    dimension so that each slice receives features from its neighbors.
    Boundary slices keep their own features (no zero-fill discontinuity).

    Args:
        x: (D, H, W, C) — BHWC format where B=num_slices
        shift_ratio: fraction of channels to shift in each direction (default 25%)

    Returns:
        (D, H, W, C) — shifted tensor.
    """
    D, H, W, C = x.shape
    n_shift = int(C * shift_ratio)

    out = x.clone()
    # Forward shift: slice i gets channels from slice i-1 (i > 0)
    out[1:, :, :, :n_shift] = x[:-1, :, :, :n_shift]
    # Backward shift: slice i gets channels from slice i+1 (i < D-1)
    out[:-1, :, :, n_shift:2 * n_shift] = x[1:, :, :, n_shift:2 * n_shift]
    # Boundary slices keep own features from clone (no zeroing)
    # Remaining channels (50%) are unchanged
    return out
