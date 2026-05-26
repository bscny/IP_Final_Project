import torch
import math
from typing import Tuple
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, min_i: float, max_i: float) -> float:
    """Calculates PSNR in dB. Assumes float tensors in range [min_i, max_i]."""
    pred = pred.clamp(min_i, max_i)  # Ensure bounds before metric calculation
    mse = ((pred - target) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10((max_i - min_i) ** 2 / mse)

def load_image_tensor(path: Path, device: torch.device, min_i: float, max_i: float) -> torch.Tensor:
    """Loads an RGB image into a float32 tensor (1, 3, H, W) and scales it exactly to the (min_i, max_i) interval."""
    img = Image.open(path).convert("RGB")
    t   = TF.to_tensor(img) # Default is [0.0, 1.0]
    
    # Scale to user-defined interval from settings.py
    t = t * (max_i - min_i) + min_i
    return t.unsqueeze(0).to(device)

def save_tensor_image(tensor: torch.Tensor, path: Path, min_i: float, max_i: float) -> None:
    """Reverts a (1, 3, H, W) tensor from (min_i, max_i) back to [0, 255] and saves to PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Normalize back to 0.0 - 1.0
    normalized = (tensor.squeeze(0) - min_i) / (max_i - min_i)
    
    # Scale to 0-255 for PNG saving
    arr = (normalized.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)

def pad_to_multiple(tensor: torch.Tensor, multiple: int = 32) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pads spatial dimensions (H, W) using reflection to be divisible by `multiple`."""
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (pad_h, pad_w)

def unpad(tensor: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    """Removes the padding applied by `pad_to_multiple`."""
    pad_h, pad_w = pad_hw
    _, _, H, W = tensor.shape
    return tensor[:, :, :H - pad_h if pad_h else H, :W - pad_w if pad_w else W]
