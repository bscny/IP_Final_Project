import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import settings
from src.unet import Noise2NoiseUNet


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class TrainConfig:
    clean_dir: str
    val_clean_dir: str | None
    output: str
    checkpoint_dir: str
    epochs: int
    batch_size: int
    crop_size: int
    noise_sigma: float
    lr: float
    weight_decay: float
    val_fraction: float
    num_workers: int
    seed: int
    save_every: int


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    paths = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths.sort(key=lambda p: str(p.relative_to(root)).lower())
    if not paths:
        raise FileNotFoundError(f"No supported image files found under: {root}")
    return paths


# trasform RGB[0, 255] to [0, 1] tensor 
def load_rgb_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


# if image < 256*256, pad the margin
def pad_if_needed(tensor: torch.Tensor, min_h: int, min_w: int) -> torch.Tensor:
    _, h, w = tensor.shape
    pad_h = max(min_h - h, 0)
    pad_w = max(min_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return tensor
    mode = "reflect" if h > pad_h and w > pad_w else "replicate" # use different strategies to pad
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)


# ensure every crop of a image is the same
def crop_pair(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    crop_size: int,
    random_crop: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if crop_size <= 0:
        return noisy, clean

    noisy = pad_if_needed(noisy, crop_size, crop_size)
    clean = pad_if_needed(clean, crop_size, crop_size)
    _, h, w = clean.shape

    if random_crop:
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
    else:
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2

    return (
        noisy[:, top:top + crop_size, left:left + crop_size],
        clean[:, top:top + crop_size, left:left + crop_size],
    )


class DIV2KDenoisingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        clean_paths: Sequence[Path],
        crop_size: int,
        noise_sigma: float,
        random_crop: bool,
    ) -> None:
        self.clean_paths = list(clean_paths)
        self.crop_size = crop_size
        self.noise_sigma = noise_sigma
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean = load_rgb_tensor(self.clean_paths[index])
        noisy = clean + torch.randn_like(clean) * (self.noise_sigma / 255.0) # 0.1
        noisy = noisy.clamp(0.0, 1.0) # restrict the noise
        noisy, clean = crop_pair(noisy, clean, self.crop_size, self.random_crop)
        return noisy, clean


def split_train_val(
    clean_paths: Sequence[Path],
    val_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    indices = list(range(len(clean_paths)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = int(round(len(indices) * val_fraction))
    n_val = min(max(n_val, 1 if len(indices) > 1 and val_fraction > 0 else 0), len(indices) - 1)
    val_indices = set(indices[:n_val])

    train_clean, val_clean = [], []

    for i, clean in enumerate(clean_paths):
        if i in val_indices:
            val_clean.append(clean)
        else:
            train_clean.append(clean)

    return train_clean, val_clean


def compute_psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    log_every: int,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_items = 0

    for step, (noisy, clean) in enumerate(loader, start=1):
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            pred = model(noisy)
            loss = criterion(pred, clean)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = noisy.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

        if is_train and (step == 1 or step % log_every == 0):
            print(
                f"[epoch {epoch:03d}] step {step:04d}/{len(loader):04d} "
                f"train_mse={loss.item():.6f} psnr={compute_psnr_from_mse(loss.item()):.2f}dB",
                flush=True,
            )

    avg_loss = total_loss / max(total_items, 1)
    return avg_loss, compute_psnr_from_mse(avg_loss)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainConfig,
    train_loss: float,
    val_loss: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised pre-training for Noise2Noise U-Net on DIV2K denoising."
    )
    parser.add_argument("--clean-dir", type=Path, default=settings.CLEAN_DIR)
    parser.add_argument("--val-clean-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=settings.OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=settings.CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--crop-size", type=int, default=settings.CROP_SIZE)
    parser.add_argument("--noise-sigma", type=float, default=settings.NOISE_SIGMA, help="Gaussian noise std in 0-255 scale.")
    parser.add_argument("--lr", type=float, default=settings.PRETRAIN_LR)
    parser.add_argument("--weight-decay", type=float, default=settings.WEIGHT_DECAY)
    parser.add_argument("--val-fraction", type=float, default=settings.VAL_FRACTION)
    parser.add_argument("--num-workers", type=int, default=settings.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=settings.SEED)
    parser.add_argument("--save-every", type=int, default=settings.SAVE_EVERY)
    parser.add_argument("--log-every", type=int, default=settings.LOG_EVERY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.crop_size > 0 and args.crop_size % 32 != 0:
        raise ValueError("--crop-size must be divisible by 32 for the current U-Net")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_clean = list_images(args.clean_dir)

    if args.val_clean_dir is not None:
        val_clean = list_images(args.val_clean_dir)
        train_clean = list(all_clean)
    else:
        train_clean, val_clean = split_train_val(
            all_clean,
            args.val_fraction,
            args.seed,
        )

    config = TrainConfig(
        clean_dir=str(args.clean_dir),
        val_clean_dir=str(args.val_clean_dir) if args.val_clean_dir else None,
        output=str(args.output),
        checkpoint_dir=str(args.checkpoint_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        noise_sigma=args.noise_sigma,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        save_every=args.save_every,
    )

    print(json.dumps(asdict(config), indent=2), flush=True)
    print(f"train images: {len(train_clean)} | val images: {len(val_clean)}", flush=True)

    train_dataset = DIV2KDenoisingDataset(
        train_clean,
        crop_size=args.crop_size,
        noise_sigma=args.noise_sigma,
        random_crop=True,
    )
    val_dataset = DIV2KDenoisingDataset(
        val_clean,
        crop_size=args.crop_size,
        noise_sigma=args.noise_sigma,
        random_crop=False,
    ) if val_clean else None

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    ) if val_dataset is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Noise2NoiseUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_psnr = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            epoch,
            args.log_every,
        )

        val_loss = None
        val_psnr = None
        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_psnr = run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device,
                    optimizer=None,
                    epoch=epoch,
                    log_every=args.log_every,
                )

        msg = (
            f"[epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_mse={train_loss:.6f} train_psnr={train_psnr:.2f}dB"
        )
        if val_loss is not None and val_psnr is not None:
            msg += f" val_mse={val_loss:.6f} val_psnr={val_psnr:.2f}dB"
        print(msg, flush=True)

        save_checkpoint(
            args.checkpoint_dir / "last.pt",
            model,
            optimizer,
            epoch,
            config,
            train_loss,
            val_loss,
        )
        if epoch % args.save_every == 0:
            save_checkpoint(
                args.checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                config,
                train_loss,
                val_loss,
            )
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                args.checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                config,
                train_loss,
                val_loss,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    elapsed = time.time() - start_time
    print(f"Saved state_dict for finetune.py: {args.output}", flush=True)
    print(f"Elapsed: {elapsed / 60.0:.1f} min", flush=True)


if __name__ == "__main__":
    main()
