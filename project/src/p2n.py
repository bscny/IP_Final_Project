import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List
import gc

# Custom Modules
from src.utils.image_helper import compute_psnr, unpad

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
        x_hat: torch.Tensor, n_hat: torch.Tensor, sigma: float, min_i: int, max_i: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs the positive/negative noisy image pair — equations (3) - (9).

    Step-by-step:
    1. Get the base denoised image x̂ and noise n̂ from the pre-trained model.
    2. Sample two *independent* positive scale factors
       σ_p, σ_n ~ N(1, σ)  (same shape as the image tensor).   eq. (5)
       Drawing them *per-pixel* (multi-scale) means the network
       sees a wide variety of noise magnitudes every iteration —
       crucial for learning to be noise-agnostic.
    3. Positive noisy image:  y_p = x̂ + σ_n · n̂              eq. (7)
    4. Negative noisy image:  y_n = x̂ − σ_p · n̂              eq. (9)
    
    Args:
        x_hat:     Base denoised image, shape (1, C, H, W), float32 ∈ [min_i, max_i]
        n_hat:     Predicted noise, shape (1, C, H, W), float32.
        sigma:     Variance of the sampled Normal Distribution

    Returns:
        two noisy images
    """
    # σ_p, σ_n sampled independently from N(1, σ), same spatial size
    # Using `torch.randn_like` gives ~ N(0,1); scaling gives N(1, σ).
    sigma_p = 1.0 + sigma * torch.randn_like(n_hat)   # σ_p ~ N(1, σ)
    sigma_n = 1.0 + sigma * torch.randn_like(n_hat)   # σ_n ~ N(1, σ)

    y_p = x_hat + sigma_n * n_hat              # positive noisy image, eq. (7)
    y_n = x_hat - sigma_p * n_hat              # negative noisy image, eq. (9)

    # Clamp to valid pixel range so the network inputs stay well-behaved.
    y_p = y_p.clamp(min_i, max_i)  # Default to [0, 1]
    y_n = y_n.clamp(min_i, max_i)  # Default to [0, 1]

    return y_p, y_n


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_p2n(
    model: nn.Module,
    x_hat: torch.Tensor,
    dirty_img: torch.Tensor,
    gt_img: torch.Tensor,
    pad_hw: Tuple[int, int],
    num_iterations: int,
    lr: float,
    sigma: float,
    gamma_start: float,
    gamma_end: float,
    log_every: int ,
    min_i: int , max_i: int,
    crop_size: int
) -> Tuple[torch.Tensor, List[int], List[float], List[float]]:
    """
    Full Positive2Negative self-supervised training loop for a *single* image.

    Args:
        model          : Pre-trained Noise2NoiseUNet.
        dirty_img      : Padded noisy input (to multiple of 32), shape (1, C, H, W), float32 ∈ [min_i, max_i].
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
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    step_history = []
    loss_history = []
    psnr_history = []
    
    # Get the predicted noise from the base model
    n_hat = dirty_img - x_hat
    
    _, _, h, w = x_hat.shape
    
    # Now, we use the n_hat and x_hat to construct the renoised pairs inside the training loop.
    for i in range(1, num_iterations + 1):
        # Linear gamma schedule
        # γ decreases linearly from `gamma_start` (2.0) → `gamma_end` (1.5)
        progress = (i - 1) / max(num_iterations - 1, 1)   # in case num_iteration is set to only 1
        gamma = gamma_start + progress * (gamma_end - gamma_start)
        
        # If the image is smaller than the crop size, use the whole image
        current_crop_h = min(crop_size, h)
        current_crop_w = min(crop_size, w)
        
        # Randomly crop a patch from the base denoised image for this iteration
        top = torch.randint(0, h - current_crop_h + 1, (1,)).item()
        left = torch.randint(0, w - current_crop_w + 1, (1,)).item()
        
        # Extract the patch
        x_hat_crop = x_hat[..., top:top+current_crop_h, left:left+current_crop_w]
        n_hat_crop = n_hat[..., top:top+current_crop_h, left:left+current_crop_w]

        # Renoised Data Construction (RDC)
        y_p, y_n = build_renoised_pair(x_hat_crop, n_hat_crop, sigma=sigma, min_i=min_i, max_i=max_i)

        # Training starts for the DCS forward passes
        model.train()
        optimizer.zero_grad()

        # Denoised Consistency Supervision (DCS)
        # Two forward passes share the same network weights.
        # The loss forces F_θ(y_p) ≈ F_θ(y_n)
        pred_pos = model(y_p)
        pred_neg = model(y_n)

        loss = p2n_loss(pred_pos, pred_neg, gamma=gamma)

        loss.backward()
        optimizer.step()

        if i % log_every == 0 or i == 1:
            step_history.append(i)
            loss_history.append(loss.item())
            
            # MEMORY FIX: Flush training tensors from VRAM
            # ==========================================
            del pred_pos, pred_neg, y_p, y_n, loss
            torch.cuda.empty_cache()

            # Evaluate PSNR for this step
            model.eval()
            with torch.no_grad():
                # Inference on the FULL image for accurate PSNR logging
                # Get intermediate prediction
                current_denoised_pad = model(dirty_img)
            
            # Unpad to match the ground truth dimensions
            current_denoised = unpad(current_denoised_pad, pad_hw)
            
            # Calculate PSNR
            current_psnr = compute_psnr(current_denoised, gt_img, min_i, max_i)
            psnr_history.append(current_psnr)

            print(f"[iter {i:4d}/{num_iterations}]  loss={loss.item():.6f}"
                  f"  γ={gamma:.4f}  PSNR={current_psnr:.4f} dB")
            
            # MEMORY FIX: Flush eval tensors to prepare for next training loop
            del current_denoised_pad, current_denoised
            gc.collect()
            torch.cuda.empty_cache()

    # ── Inference: one clean forward pass ───────────────────────────────
    model.eval()
    with torch.no_grad():
        denoised = model(dirty_img).clamp(min_i, max_i)

    return denoised, step_history, loss_history, psnr_history
