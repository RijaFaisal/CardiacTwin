"""
core/inference/fcn_wang_gradcam.py

Grad-CAM for the FCN Wang PyTorch model trained on PTB-XL.

Hooks the last Conv1D layer (block 2, 256→128 filters) to produce a
(1000,) temporal saliency map showing which time points in the full
10-second ECG most influenced the top predicted diagnosis.

This explains the production model's actual decisions — unlike the
MIT-BIH CNN Grad-CAM which only covers individual beats.

Reference: Selvaraju et al. "Grad-CAM" (ICCV 2017).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import zoom


def compute_fcn_wang_gradcam(
    model: torch.nn.Module,
    ecg_scaled: np.ndarray,
    class_idx: int,
    target_length: int = 1000,
) -> list[float]:
    """
    Compute a temporal Grad-CAM heatmap for one ECG record.

    Parameters
    ----------
    model        : FCN Wang model already in eval() mode with weights loaded
    ecg_scaled   : (1000, 12) float32 — preprocessed, scaled ECG signal
    class_idx    : which of the 71 PTB-XL classes to explain (highest-prob class)
    target_length: output length; should match ecg_scaled rows (1000)

    Returns
    -------
    list[float]: saliency values in [0, 1], length == target_length.
        Values near 1 = time points that most drove the predicted diagnosis.
    """
    activations: dict[str, torch.Tensor] = {}
    gradients:   dict[str, torch.Tensor] = {}

    # Find last Conv1d layer dynamically — robust to architecture changes
    last_conv: torch.nn.Conv1d | None = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            last_conv = module

    if last_conv is None:
        return [0.0] * target_length

    fwd_hook = last_conv.register_forward_hook(
        lambda _m, _i, out: activations.update({"v": out})
    )
    bwd_hook = last_conv.register_full_backward_hook(
        lambda _m, _gi, go: gradients.update({"v": go[0]})
    )

    try:
        # (1000, 12) → transpose → (12, 1000) → batch dim → (1, 12, 1000)
        x = torch.tensor(ecg_scaled.T[np.newaxis], dtype=torch.float32)

        model.zero_grad()
        logits = model(x)                          # (1, 71)
        score  = torch.sigmoid(logits)[0, class_idx]
        score.backward()

        acts  = activations["v"]                   # (1, C, T)
        grads = gradients["v"]                     # (1, C, T)

        # Global-average-pool gradients across time → one weight per filter
        weights = grads.mean(dim=2, keepdim=True)  # (1, C, 1)
        cam = (weights * acts).sum(dim=1).squeeze(0)  # (T,)
        cam = torch.relu(cam).detach().numpy()

        # Upsample from conv output length (~999) back to target_length
        conv_len = len(cam)
        if conv_len != target_length:
            cam = zoom(cam, target_length / conv_len)[:target_length]

        cam = np.clip(cam, 0, None)
        mx  = cam.max()
        if mx > 1e-8:
            cam = cam / mx

        return cam.tolist()

    finally:
        fwd_hook.remove()
        bwd_hook.remove()
