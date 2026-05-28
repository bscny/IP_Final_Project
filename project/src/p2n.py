import torch.nn as nn
import torch.optim as optim
import torch
from typing import Tuple, List

# Custom Modules
from src.utils.image_helper import compute_psnr, unpad


def _validate_multiple(name: str, value: int, multiple: int = 32) -> None:
    if value <= 0 or value % multiple != 0:
        raise ValueError(f"{name} must be a positive multiple of {multiple}; got {value}")


def random_spatial_crop(tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    Returns a random square crop. If the image is smaller than the requested crop,
    returns the original tensor.
    """
    if crop_size <= 0:
        return tensor

    _, _, height, width = tensor.shape
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)

    if crop_h == height and crop_w == width:
        return tensor

    top = torch.randint(0, height - crop_h + 1, (1,), device=tensor.device).item()
    left = torch.randint(0, width - crop_w + 1, (1,), device=tensor.device).item()
    return tensor[:, :, top:top + crop_h, left:left + crop_w]


def _tile_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def denoise_tiled(
    model: nn.Module,
    image: torch.Tensor,
    tile_size: int,
    overlap: int,
    min_i: float,
    max_i: float,
) -> torch.Tensor:
    """
    Runs full-image denoising in overlapping tiles to keep inference memory
    bounded for high-resolution benchmark images.
    """
    _validate_multiple("tile_size", tile_size)
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"overlap must be in [0, tile_size); got {overlap}")

    _, _, height, width = image.shape
    if height <= tile_size and width <= tile_size:
        return model(image).clamp(min_i, max_i)

    tile_h = min(tile_size, height)
    tile_w = min(tile_size, width)
    stride = tile_size - overlap
    y_starts = _tile_starts(height, tile_h, stride)
    x_starts = _tile_starts(width, tile_w, stride)

    output = torch.zeros_like(image)
    counts = torch.zeros((image.shape[0], 1, height, width), device=image.device, dtype=image.dtype)

    for top in y_starts:
        bottom = top + tile_h
        for left in x_starts:
            right = left + tile_w
            tile = image[:, :, top:bottom, left:right]
            output[:, :, top:bottom, left:right] += model(tile).clamp(min_i, max_i)
            counts[:, :, top:bottom, left:right] += 1

    return output / counts.clamp_min(1)

# ─────────────────────────────────────────────────────────────────────────────
# Denoised Consistency Supervision (DCS)
# ─────────────────────────────────────────────────────────────────────────────
def p2n_loss(pred_pos: torch.Tensor, pred_neg: torch.Tensor, gamma: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Args:
        pred_pos: F_θ(y_p)  — network output for the *positive* noisy image.
        pred_neg: F_θ(y_n)  — network output for the *negative* noisy image.
        gamma:    current exponent γ ∈ [1.5, 2.0].
        eps:      small constant for numerical stability (prevents log(0)).

    Returns:
        Scalar loss value.
    """
    diff = pred_pos - pred_neg                  # element-wise residual
    loss = (diff.abs() + eps).pow(gamma)        # (|x| + ε)^γ,  eq. (11)
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Renoised Data Construction (RDC)
# ─────────────────────────────────────────────────────────────────────────────
def build_renoised_pair(
        model: nn.Module, noisy_img: torch.Tensor, sigma: float, min_i: int, max_i: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs the positive/negative noisy image pair — equations (3) - (9).

    Step-by-step:

    1. One forward pass gives the predicted clean image x̂ = F_θ(y).
    2. Predicted noise:  n̂ = y - x̂                            eq. (4)
    3. Sample two *independent* positive scale factors
       σ_p, σ_n ~ N(1, σ)  (same shape as the image tensor).   eq. (5)
       Drawing them *per-pixel* (multi-scale) means the network
       sees a wide variety of noise magnitudes every iteration —
       crucial for learning to be noise-agnostic.
    4. Positive noisy image:  y_p = x̂ + σ_n · n̂              eq. (7)
    5. Negative noisy image:  y_n = x̂ − σ_p · n̂              eq. (9)
    
    Args:
        model:     Pre-trained Noise2NoiseUNet.
        noisy_img: Noisy input, shape (1, C, H, W), float32 ∈ [min_i, max_i].
        sigma:     Variance of the sampled Normal Distribution

    Returns:
        two noisy images
    """
    # Don't track gradients for inferencing
    model.eval()
    with torch.no_grad():
        x_hat = model(noisy_img)

    n_hat = noisy_img - x_hat

    # σ_p, σ_n sampled independently from N(1, σ), same spatial size
    # Using `torch.randn_like` gives ~ N(0,1); scaling gives N(1, σ).
    sigma_p = 1.0 + sigma * torch.randn_like(n_hat)   # σ_p ~ N(1, σ)
    sigma_n = 1.0 + sigma * torch.randn_like(n_hat)   # σ_n ~ N(1, σ)

    y_p = x_hat + sigma_n * n_hat              # positive noisy image, eq. (7)
    y_n = x_hat - sigma_p * n_hat              # negative noisy image, eq. (9)

    # Clamp to valid pixel range so the network inputs stay well-behaved.
    y_p = y_p.clamp(min_i, max_i)  # Default to [0, 255]
    y_n = y_n.clamp(min_i, max_i)  # Default to [0, 255]

    return y_p, y_n


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_p2n(
    model: nn.Module,
    dirty_img: torch.Tensor,
    gt_img: torch.Tensor,
    pad_hw: Tuple[int, int],
    num_iterations: int,
    lr: float,
    sigma: float,
    gamma_start: float,
    gamma_end: float,
    log_every: int ,
    min_i: int,
    max_i: int,
    crop_size: int = 1024,
    tile_size: int = 1024,
    tile_overlap: int = 64,
) -> Tuple[torch.Tensor, List[int], List[float], List[float]]:
    """
    Full Positive2Negative self-supervised training loop for a *single* image.

    Args:
        model          : Pre-trained Noise2NoiseUNet.
        dirty_img      : Noisy input, shape (1, C, H, W), float32 ∈ [min_i, max_i].
        num_iterations : The paper converges in ~100 iterations on SIDD
        lr             : AdamW learning rate (paper uses 1e-4).
        sigma          : Spread of the scale-parameter distribution N(1, σ).
                         Paper fixes σ = 0.75 for all experiments.
        gamma_start    : Initial norm exponent (ℓ_2, least sensitive to
                         outliers, ideal for unstable early training).
        gamma_end      : Final norm exponent (ℓ_1.5, heavier-tailed,
                         more robust to real-world noise).
        log_every      : Print loss every N iterations.

    Returns:
        denoised     : The final denoised image tensor (1, C, H, W), detached,
                       clipped to [min_i, max_i], on the same device as `dirty_img`.
        step_history : The step according to log_every
        loss_history : The loss according to log_every
        psnr_history : The psnr according to log_every
    """
    if crop_size > 0:
        _validate_multiple("crop_size", crop_size)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    step_history = []
    loss_history = []
    psnr_history = []

    for i in range(1, num_iterations + 1):
        # Linear gamma schedule
        # γ decreases linearly from `gamma_start` (2.0) → `gamma_end` (1.5)
        progress = (i - 1) / max(num_iterations - 1, 1)   # in case num_iteration is set to only 1
        gamma = gamma_start + progress * (gamma_end - gamma_start)

        dirty_patch = random_spatial_crop(dirty_img, crop_size)

        # Renoised Data Construction (RDC)
        y_p, y_n = build_renoised_pair(model, dirty_patch, sigma=sigma, min_i=min_i, max_i=max_i)

        # Switch back to training mode for the DCS forward passes
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Denoised Consistency Supervision (DCS)
        # Two forward passes share the same network weights.
        # The loss forces F_θ(y_p) ≈ F_θ(y_n)
        pred_pos = model(y_p)
        pred_neg = model(y_n)

        loss = p2n_loss(pred_pos, pred_neg, gamma=gamma)

        loss.backward()
        optimizer.step()
        loss_value = loss.item()

        del dirty_patch, y_p, y_n, pred_pos, pred_neg, loss

        if i % log_every == 0 or i == 1:
            step_history.append(i)
            loss_history.append(loss_value)

            # Evaluate PSNR for this step
            model.eval()
            with torch.no_grad():
                # Get intermediate prediction
                current_denoised_pad = denoise_tiled(
                    model,
                    dirty_img,
                    tile_size=tile_size,
                    overlap=tile_overlap,
                    min_i=min_i,
                    max_i=max_i,
                )
            
            # Unpad to match the ground truth dimensions
            current_denoised = unpad(current_denoised_pad, pad_hw)
            
            # Calculate PSNR
            current_psnr = compute_psnr(current_denoised, gt_img, min_i, max_i)
            psnr_history.append(current_psnr)

            print(f"[iter {i:4d}/{num_iterations}]  loss={loss_value:.6f}"
                  f"  γ={gamma:.4f}  PSNR={current_psnr:.4f} dB")

            del current_denoised_pad, current_denoised

    # ── Inference: one clean forward pass ───────────────────────────────
    model.eval()
    with torch.no_grad():
        denoised = denoise_tiled(
            model,
            dirty_img,
            tile_size=tile_size,
            overlap=tile_overlap,
            min_i=min_i,
            max_i=max_i,
        )

    return denoised, step_history, loss_history, psnr_history
