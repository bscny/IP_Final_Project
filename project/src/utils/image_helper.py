import torch
import math
from typing import Tuple
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

def compute_psnr(pred: torch.Tensor, target: torch.Tensor, min_i: float, max_i: float) -> float:
    """Calculates PSNR in dB. Assumes float tensors in range [min_i, max_i]."""
    pred = pred.clamp(min_i, max_i)      # Ensure bounds before metric calculation
    target = target.clamp(min_i, max_i)  # Ensure bounds before metric calculation
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

def get_tiled_prediction(model: torch.nn.Module, img: torch.Tensor, tile_size: int = 1024) -> torch.Tensor:
    """
    Processes a massive image in smaller tiles to prevent VRAM explosion during inference.
    """
    b, c, h, w = img.shape
    out = torch.zeros_like(img)
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            # Calculate tile boundaries
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            # Extract the tile
            tile = img[..., y:y_end, x:x_end]
            
            # Pad the tile to a multiple of 32 for the UNet
            # (Assuming pad_to_multiple and unpad are imported in p2n.py)
            tile_pad, pad_hw = pad_to_multiple(tile, 32)
            
            # Forward pass only on the small tile
            with torch.no_grad():
                pred_pad = model(tile_pad)
                
            # Unpad and slot back into the final output tensor
            pred = unpad(pred_pad, pad_hw)
            out[..., y:y_end, x:x_end] = pred
            
            # Keep VRAM clean between tiles
            del tile, tile_pad, pred_pad, pred
            torch.cuda.empty_cache()
            
    return out
