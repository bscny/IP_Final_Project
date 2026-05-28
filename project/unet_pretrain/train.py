import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Custom Modules
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
    use_padding: bool  # the padding toggle


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


# trasform RGB[0, 255] to [MIN_I, MAX_I] tensor 
def load_rgb_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    # Normalize to [0, 1] first
    arr = np.asarray(img, dtype=np.float32) / 255.0
    
    # Scale and shift to the target range [MIN_I, MAX_I]
    data_range = settings.MAX_I - settings.MIN_I
    arr = (arr * data_range) + settings.MIN_I
    
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


# if image < target padded size, pad the margin
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


# Dynamic Batch Padding Collate Function
def pad_collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamically pads a batch of arbitrarily sized images to the max 
    dimension found in the batch, rounded up to the nearest multiple of 32.
    """
    max_h = max(item[0].shape[1] for item in batch)
    max_w = max(item[0].shape[2] for item in batch)
    
    # Round up to nearest multiple of 32 for U-Net compatibility
    target_h = (max_h + 31) // 32 * 32
    target_w = (max_w + 31) // 32 * 32
    
    noisy_padded = [pad_if_needed(n, target_h, target_w) for n, _ in batch]
    clean_padded = [pad_if_needed(c, target_h, target_w) for _, c in batch]
    
    return torch.stack(noisy_padded), torch.stack(clean_padded)


class DIV2KDenoisingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        clean_paths: Sequence[Path],
        crop_size: int,
        noise_sigma: float,
        random_crop: bool,
        use_padding: bool = False, # toggle parameter
    ) -> None:
        self.clean_paths = list(clean_paths)
        self.crop_size = crop_size
        self.noise_sigma = noise_sigma
        self.random_crop = random_crop
        self.use_padding = use_padding

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean = load_rgb_tensor(self.clean_paths[index])
    
        # Scale the noise_sigma (0-255 scale) to your custom dynamic range
        range_scale = (settings.MAX_I - settings.MIN_I) / 255.0
        scaled_sigma = self.noise_sigma * range_scale
        
        noisy = clean + torch.randn_like(clean) * scaled_sigma
        
        # Clamp the noise to the new min/max bounds
        noisy = noisy.clamp(settings.MIN_I, settings.MAX_I) 
        
        # Bypass crop if using the padding method
        if not self.use_padding:
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
    # Calculate the dynamic range (R)
    dynamic_range = settings.MAX_I - settings.MIN_I
    
    # Standard PSNR formula using the squared dynamic range
    return 10.0 * math.log10((dynamic_range ** 2) / mse)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed((seed + worker_id) % (2**32))
    np.random.seed((seed + worker_id) % (2**32))


def init_distributed() -> tuple[bool, int, int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1, settings.DEVICE

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)
    return True, rank, local_rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def main_print(rank: int, *args, **kwargs) -> None:
    if is_main_process(rank):
        print(*args, **kwargs)


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    log_every: int,
    rank: int = 0,
    distributed: bool = False,
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

        if is_train and is_main_process(rank) and (step == 1 or step % log_every == 0):
            print(
                f"[epoch {epoch:03d}] step {step:04d}/{len(loader):04d} "
                f"train_mse={loss.item():.6f} psnr={compute_psnr_from_mse(loss.item()):.2f}dB",
                flush=True,
            )

    if distributed:
        totals = torch.tensor([total_loss, float(total_items)], device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss = totals[0].item()
        total_items = int(totals[1].item())

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
    # Safely pull WORLD_SIZE from torchrun again (defaults to 1 if not running distributed)
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Divide global batch size by GPUs to get the local per-GPU batch size
    local_batch_size = max(1, settings.BATCH_SIZE // world_size)
    
    parser = argparse.ArgumentParser(
        description="Supervised pre-training for Noise2Noise U-Net on DIV2K denoising."
    )
    parser.add_argument("--clean-dir", type=Path, default=settings.CLEAN_DIR)
    parser.add_argument("--val-clean-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=settings.OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=settings.CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=settings.EPOCHS)
    
    # 3. Inject the calculated local batch size here
    parser.add_argument("--batch-size", type=int, default=local_batch_size)
    
    parser.add_argument("--crop-size", type=int, default=settings.CROP_SIZE)
    parser.add_argument("--noise-sigma", type=float, default=settings.NOISE_SIGMA, help="Gaussian noise std in 0-255 scale.")
    parser.add_argument("--lr", type=float, default=settings.PRETRAIN_LR)
    parser.add_argument("--weight-decay", type=float, default=settings.WEIGHT_DECAY)
    parser.add_argument("--val-fraction", type=float, default=settings.VAL_FRACTION)
    parser.add_argument("--num-workers", type=int, default=settings.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=settings.SEED)
    parser.add_argument("--save-every", type=int, default=settings.SAVE_EVERY)
    parser.add_argument("--log-every", type=int, default=settings.LOG_EVERY)
    
    # Add toggle to switch between padding and cropping
    parser.add_argument("--use-padding", action="store_true", default=settings.USE_PADDING, help="Pad dynamically to multiple of 32 instead of cropping.")
    
    parser.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="ip-final-project")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT.parent / ".env")
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def init_wandb(args: argparse.Namespace, config: TrainConfig):
    if not args.wandb:
        return None

    load_dotenv(args.env_file)
    import wandb
    return wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=asdict(config))


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size, device = init_distributed()
    if not args.use_padding and args.crop_size > 0 and args.crop_size % 32 != 0:
        raise ValueError("--crop-size must be divisible by 32 for the current U-Net")

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        batch_size=args.batch_size * world_size,
        crop_size=args.crop_size,
        noise_sigma=args.noise_sigma,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        save_every=args.save_every,
        use_padding=args.use_padding, # Log the state of padding
    )

    main_print(rank, json.dumps(asdict(config), indent=2), flush=True)
    main_print(
        rank,
        f"train images: {len(train_clean)} | val images: {len(val_clean)} | "
        f"world size: {world_size}",
        flush=True,
    )
    wandb_run = init_wandb(args, config) if is_main_process(rank) else None

    train_dataset = DIV2KDenoisingDataset(
        train_clean,
        crop_size=args.crop_size,
        noise_sigma=args.noise_sigma,
        random_crop=True,
        use_padding=args.use_padding, 
    )
    val_dataset = DIV2KDenoisingDataset(
        val_clean,
        crop_size=args.crop_size,
        noise_sigma=args.noise_sigma,
        random_crop=False,
        use_padding=args.use_padding,
    ) if (val_clean and is_main_process(rank)) else None

    # Conditionally attach the custom collate_fn
    collate_fn = pad_collate_fn if args.use_padding else None

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    ) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=generator,
        collate_fn=collate_fn, # Apply custom batching
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn, # Apply custom batching
    ) if val_dataset is not None and is_main_process(rank) else None

    raw_model = Noise2NoiseUNet().to(device)
    model: nn.Module = raw_model
    if distributed:
        model = DistributedDataParallel(
            raw_model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
        )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_psnr = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            epoch,
            args.log_every,
            rank=rank,
            distributed=distributed,
        )

        val_loss = None
        val_psnr = None
        if distributed:
            dist.barrier()
        if val_loader is not None and is_main_process(rank):
            with torch.no_grad():
                val_loss, val_psnr = run_epoch(
                    raw_model,
                    val_loader,
                    criterion,
                    device,
                    optimizer=None,
                    epoch=epoch,
                    log_every=args.log_every,
                    rank=rank,
                    distributed=False,
                )
        if distributed:
            dist.barrier()

        if is_main_process(rank):
            msg = (
                f"[epoch {epoch:03d}/{args.epochs:03d}] "
                f"train_mse={train_loss:.6f} train_psnr={train_psnr:.2f}dB"
            )
            if val_loss is not None and val_psnr is not None:
                msg += f" val_mse={val_loss:.6f} val_psnr={val_psnr:.2f}dB"
            print(msg, flush=True)

            save_checkpoint(
                args.checkpoint_dir / "last.pt",
                raw_model,
                optimizer,
                epoch,
                config,
                train_loss,
                val_loss,
            )
            if epoch % args.save_every == 0:
                save_checkpoint(
                    args.checkpoint_dir / f"epoch_{epoch:03d}.pt",
                    raw_model,
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
                    raw_model,
                    optimizer,
                    epoch,
                    config,
                    train_loss,
                    val_loss,
                )
            if wandb_run is not None:
                metrics = {
                    "epoch": epoch,
                    "train/mse": train_loss,
                    "train/psnr": train_psnr,
                    "elapsed_minutes": (time.time() - start_time) / 60.0,
                }
                if val_loss is not None and val_psnr is not None:
                    metrics["val/mse"] = val_loss
                    metrics["val/psnr"] = val_psnr
                    metrics["val/best_mse"] = best_val_loss
                wandb_run.log(metrics)

    if is_main_process(rank):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(raw_model.state_dict(), args.output)
        elapsed = time.time() - start_time
        print(f"Saved state_dict for finetune.py: {args.output}", flush=True)
        print(f"Elapsed: {elapsed / 60.0:.1f} min", flush=True)
        if wandb_run is not None:
            wandb_run.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
