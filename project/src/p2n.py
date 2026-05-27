import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List

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
    min_i: int , max_i: int
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

        # Renoised Data Construction (RDC)
        y_p, y_n = build_renoised_pair(model, dirty_img, sigma=sigma, min_i=min_i, max_i=max_i)

        # Switch back to training mode for the DCS forward passes
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

            # Evaluate PSNR for this step
            model.eval()
            with torch.no_grad():
                # Get intermediate prediction
                current_denoised_pad = model(dirty_img)
            
            # Unpad to match the ground truth dimensions
            current_denoised = unpad(current_denoised_pad, pad_hw)
            
            # Calculate PSNR
            current_psnr = compute_psnr(current_denoised, gt_img, min_i, max_i)
            psnr_history.append(current_psnr)

            print(f"[iter {i:4d}/{num_iterations}]  loss={loss.item():.6f}"
                  f"  γ={gamma:.4f}  PSNR={current_psnr:.4f} dB")

    # ── Inference: one clean forward pass ───────────────────────────────
    model.eval()
    with torch.no_grad():
        denoised = model(dirty_img).clamp(min_i, max_i)

    return denoised, step_history, loss_history, psnr_history
