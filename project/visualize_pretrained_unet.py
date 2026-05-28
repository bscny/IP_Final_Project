import argparse
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

import settings
from src.p2n import denoise_tiled
from src.unet import Noise2NoiseUNet
from src.utils.data_helper import find_cc_pairs, find_polyu_pairs, find_sidd_pairs
from src.utils.image_helper import (
    compute_psnr,
    load_image_tensor,
    pad_to_multiple,
    save_tensor_image,
    unpad,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize pretrained U-Net denoising results and difference images."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to pretrained U-Net .pt file. If omitted, common project paths are tried.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.DATA_DIR,
        help="Dataset root. Defaults to settings.DATA_DIR.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Single noisy image to run. If omitted, known datasets under data-dir are scanned.",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Optional ground-truth image for --image.",
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "CC", "PolyU", "SIDD"),
        default="all",
        help="Dataset subset to scan when --image is not provided.",
    )
    parser.add_argument("--limit", type=int, default=12, help="Maximum number of images to process.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=settings.RESULT_DIR / "pretrained_visuals",
        help="Output directory for generated comparison images.",
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument(
        "--diff-gain",
        type=float,
        default=4.0,
        help="Multiplier for visualizing absolute difference images.",
    )
    return parser.parse_args()


def resolve_weights(path: Path | None) -> Path:
    candidates = []
    if path is not None:
        candidates.append(path)
    candidates.extend(
        [
            Path("pretrain_unet.pt"),
            Path("pretrained_unet.pt"),
            Path("unet_pretrain.pt"),
            Path("unet_pretrain/unet_pretrain.pt"),
            settings.WEIGHTS_PATH,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"No pretrained weights found. Tried:\n{tried}")


def load_state_dict(weights_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {weights_path}")

    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }


def tensor_to_pil(tensor: torch.Tensor, min_i: float, max_i: float) -> Image.Image:
    tensor = tensor.detach().float().cpu().squeeze(0)
    tensor = ((tensor - min_i) / (max_i - min_i)).clamp(0, 1)
    array = (tensor * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def diff_to_pil(
    left: torch.Tensor,
    right: torch.Tensor,
    min_i: float,
    max_i: float,
    gain: float,
) -> Image.Image:
    diff = (left.detach() - right.detach()).abs() * gain
    return tensor_to_pil(diff + min_i, min_i, max_i)


def draw_label(image: Image.Image, label: str) -> Image.Image:
    labeled = Image.new("RGB", (image.width, image.height + 28), "white")
    labeled.paste(image.convert("RGB"), (0, 28))

    draw = ImageDraw.Draw(labeled)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, labeled.width, 27), fill=(245, 245, 245))
    draw.text((8, 8), label, fill=(20, 20, 20), font=font)
    return labeled


def make_contact_sheet(columns: list[tuple[str, Image.Image]], max_width: int = 360) -> Image.Image:
    labeled = []
    for label, image in columns:
        scale = min(max_width / image.width, max_width / image.height, 1.0)
        if scale < 1.0:
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        labeled.append(draw_label(image, label))

    gap = 8
    width = sum(image.width for image in labeled) + gap * (len(labeled) - 1)
    height = max(image.height for image in labeled)
    sheet = Image.new("RGB", (width, height), "white")

    x = 0
    for image in labeled:
        sheet.paste(image, (x, 0))
        x += image.width + gap
    return sheet


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def discover_pairs(data_dir: Path, dataset: str) -> list[tuple[str, Path, Path | None]]:
    pairs: list[tuple[str, Path, Path | None]] = []

    if dataset in ("all", "CC") and (data_dir / "CC").exists():
        pairs.extend(("CC", noisy, gt) for noisy, gt in find_cc_pairs(data_dir / "CC"))
    if dataset in ("all", "PolyU") and (data_dir / "PolyU").exists():
        pairs.extend(("PolyU", noisy, gt) for noisy, gt in find_polyu_pairs(data_dir / "PolyU"))
    if dataset in ("all", "SIDD") and (data_dir / "SIDD").exists():
        pairs.extend(("SIDD", noisy, gt) for noisy, gt in find_sidd_pairs(data_dir / "SIDD"))

    if pairs:
        return pairs

    image_paths = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return [("images", path, None) for path in image_paths]


def run_one(
    model: Noise2NoiseUNet,
    noisy_path: Path,
    gt_path: Path | None,
    out_dir: Path,
    tile_size: int,
    tile_overlap: int,
    diff_gain: float,
) -> None:
    noisy = load_image_tensor(noisy_path, settings.DEVICE, settings.MIN_I, settings.MAX_I)
    noisy_pad, pad_hw = pad_to_multiple(noisy, 32)

    with torch.no_grad():
        denoised_pad = denoise_tiled(
            model,
            noisy_pad,
            tile_size=tile_size,
            overlap=tile_overlap,
            min_i=settings.MIN_I,
            max_i=settings.MAX_I,
        )
    denoised = unpad(denoised_pad, pad_hw)

    gt = None
    if gt_path is not None:
        gt = load_image_tensor(gt_path, settings.DEVICE, settings.MIN_I, settings.MAX_I)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(noisy_path)
    save_tensor_image(noisy, out_dir / f"{stem}_noisy.png", settings.MIN_I, settings.MAX_I)
    save_tensor_image(denoised, out_dir / f"{stem}_pretrained.png", settings.MIN_I, settings.MAX_I)

    columns = [
        ("noisy", tensor_to_pil(noisy, settings.MIN_I, settings.MAX_I)),
        ("pretrained", tensor_to_pil(denoised, settings.MIN_I, settings.MAX_I)),
        (f"abs diff x{diff_gain:g}", diff_to_pil(noisy, denoised, settings.MIN_I, settings.MAX_I, diff_gain)),
    ]

    msg = f"{noisy_path}"
    if gt is not None:
        psnr_noisy = compute_psnr(noisy, gt, settings.MIN_I, settings.MAX_I)
        psnr_pretrained = compute_psnr(denoised, gt, settings.MIN_I, settings.MAX_I)
        save_tensor_image(gt, out_dir / f"{stem}_gt.png", settings.MIN_I, settings.MAX_I)
        columns.insert(2, ("gt", tensor_to_pil(gt, settings.MIN_I, settings.MAX_I)))
        columns.append(
            (
                f"error x{diff_gain:g}",
                diff_to_pil(denoised, gt, settings.MIN_I, settings.MAX_I, diff_gain),
            )
        )
        msg += f" | noisy_psnr={psnr_noisy:.2f}dB pretrained_psnr={psnr_pretrained:.2f}dB"

    sheet = make_contact_sheet(columns)
    sheet_path = out_dir / f"{stem}_compare.png"
    sheet.save(sheet_path)
    print(f"{msg}\n  saved {sheet_path}")

    del noisy, noisy_pad, denoised, denoised_pad, gt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    weights_path = resolve_weights(args.weights)
    print(f"weights: {weights_path}")
    print(f"device: {settings.DEVICE}")

    model = Noise2NoiseUNet().to(settings.DEVICE)
    model.load_state_dict(load_state_dict(weights_path), strict=True)
    model.eval()

    if args.image is not None:
        items = [("single", args.image, args.gt)]
    else:
        items = discover_pairs(args.data_dir, args.dataset)

    if args.limit > 0:
        items = items[: args.limit]

    if not items:
        raise FileNotFoundError(f"No images found under {args.data_dir}")

    with torch.no_grad():
        for dataset_name, noisy_path, gt_path in items:
            run_one(
                model=model,
                noisy_path=noisy_path,
                gt_path=gt_path,
                out_dir=args.out_dir / dataset_name,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                diff_gain=args.diff_gain,
            )


if __name__ == "__main__":
    main()
