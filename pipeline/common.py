"""
Shared utilities for the diagnostics/ scripts.

Note (2026-09-03): reconstructed from the conversation record after
diagnostics/ was found deleted from the working tree partway through this
session (not git-tracked, so no history to recover from). See the chat for
the full report.

This module is new code (not a copy of anything in the main repo). It only
depends on torch/numpy/PIL/skimage, which are already required by
environment.yml.
"""
import os
import csv
import random

import numpy as np
import torch
from PIL import Image


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    img = img.detach().cpu().clamp(0.0, 1.0)
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def load_image_as_tensor(path: str, resolution: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((resolution, resolution), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 4):
    """images: (N, C, H, W) tensor in [0, 1]."""
    ensure_dir(os.path.dirname(path))
    n, c, h, w = images.shape
    ncol = nrow
    nrow_grid = int(np.ceil(n / ncol))
    grid = np.ones((nrow_grid * h, ncol * w, c), dtype=np.uint8) * 255
    for idx in range(n):
        r, col = divmod(idx, ncol)
        img = images[idx].detach().cpu().clamp(0.0, 1.0)
        arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = arr
    mode = "RGB" if c == 3 else "L"
    if c == 1:
        grid = grid[:, :, 0]
    Image.fromarray(grid, mode=mode).save(path)


def save_side_by_side(img_a: torch.Tensor, img_b: torch.Tensor, path: str, labels=None):
    """img_a, img_b: (C, H, W) tensors in [0, 1]. Saves them side by side."""
    ensure_dir(os.path.dirname(path))
    c, h, w = img_a.shape
    pad = 4
    canvas = np.ones((h, w * 2 + pad, c), dtype=np.uint8) * 255
    a = (img_a.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    b = (img_b.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    canvas[:, :w] = a
    canvas[:, w + pad:] = b
    Image.fromarray(canvas).save(path)


def mean_saturation(img: torch.Tensor) -> float:
    """img: (C, H, W) tensor in [0, 1], RGB. Returns mean of HSV 'S' channel in [0, 1]."""
    arr = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    s = np.zeros_like(maxc)
    nonzero = maxc > 1e-8
    s[nonzero] = (maxc[nonzero] - minc[nonzero]) / maxc[nonzero]
    return float(s.mean())


def laplacian_variance(img: torch.Tensor) -> float:
    """img: (C, H, W) tensor in [0, 1]. Grayscale Laplacian variance (high-frequency energy proxy)."""
    arr = img.detach().cpu().clamp(0, 1).numpy()
    gray = 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="reflect")
    out = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            if kernel[i, j] == 0:
                continue
            out += kernel[i, j] * padded[i:i + h, j:j + w]
    return float(out.var())


def psnr(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    """img_a, img_b: (C, H, W) tensors in [0, 1]."""
    try:
        from skimage.metrics import peak_signal_noise_ratio
        a = img_a.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        b = img_b.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        return float(peak_signal_noise_ratio(a, b, data_range=1.0))
    except ImportError:
        a = img_a.detach().cpu().clamp(0, 1).numpy()
        b = img_b.detach().cpu().clamp(0, 1).numpy()
        mse = float(np.mean((a - b) ** 2))
        if mse == 0:
            return float("inf")
        return 10.0 * np.log10(1.0 / mse)


def ssim(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    """img_a, img_b: (C, H, W) tensors in [0, 1].

    scikit-image's structural_similarity() renamed its multichannel-image
    argument from `multichannel=True` (<=0.18, which this project's
    environment.yml pins: scikit-image==0.18.3) to `channel_axis=<axis>`
    (>=0.19). 0.18.3's signature has a **kwargs catch-all, so passing
    channel_axis to it doesn't raise TypeError -- it's silently swallowed
    and the image is treated as single-channel 3D data (H, W, C-as-spatial),
    which fails with ValueError("win_size exceeds image extent") as soon as
    C < the default win_size (7). Since that failure mode isn't a clean,
    safely-catchable TypeError, inspect the installed function's signature
    up front instead of relying on exception handling.
    """
    import inspect
    from skimage.metrics import structural_similarity
    a = img_a.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    b = img_b.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    if "channel_axis" in inspect.signature(structural_similarity).parameters:
        return float(structural_similarity(a, b, data_range=1.0, channel_axis=2))
    else:
        return float(structural_similarity(a, b, data_range=1.0, multichannel=True))


def write_csv(path: str, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
