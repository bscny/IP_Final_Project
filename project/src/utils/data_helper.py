from typing import Tuple, List
from pathlib import Path
import re

def find_cc_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Matches Cross-Channel (CC) dataset pairs: *_real.png (Noisy) / *_mean.png (GT)."""
    noisy_files = sorted(root.glob("*_real.png"), key=lambda p: p.name.lower())
    pairs = []
    for noisy in noisy_files:
        stem       = re.sub(r"_real$", "", noisy.stem, flags=re.IGNORECASE)
        candidates = list(root.glob(f"{stem}_mean.png")) + list(root.glob(f"{stem}_mean.PNG"))
        if candidates:
            pairs.append((noisy, candidates[0]))
        else:
            print(f"  [CC] WARNING: No Ground Truth found for {noisy.name}")
    return pairs

def find_polyu_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Matches PolyU dataset pairs: *_Real.JPG (Noisy) / *_mean.JPG (GT)."""
    noisy_files = sorted(
        [p for p in root.iterdir() if re.search(r"_real\.(jpg|jpeg)$", p.name, re.IGNORECASE)],
        key=lambda p: p.name.lower(),
    )
    pairs = []
    for noisy in noisy_files:
        stem       = re.sub(r"_real$", "", noisy.stem, flags=re.IGNORECASE)
        suffix     = noisy.suffix
        # candidates = list(root.glob(f"{stem}_mean{suffix}")) + list(root.glob(f"{stem}_mean.JPG")) + list(root.glob(f"{stem}_mean.jpg"))
        candidates = list({p for ext in (suffix, ".JPG", ".jpg") for p in root.glob(f"{stem}_mean{ext}")})
        if candidates:
            pairs.append((noisy, candidates[0]))
        else:
            print(f"  [PolyU] WARNING: No Ground Truth found for {noisy.name}")
    return pairs

def find_sidd_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """Matches SIDD dataset pairs inside scene subdirectories: NOISY_SRGB*.PNG / GT_SRGB*.PNG."""
    pairs = []
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        noisy_files = sorted(scene_dir.glob("NOISY_SRGB_*.PNG"))
        gt_files    = sorted(scene_dir.glob("GT_SRGB_*.PNG"))
        
        if not noisy_files or not gt_files:
            print(f"  [SIDD] WARNING: Incomplete scene {scene_dir.name}")
            continue
            
        pairs.append((noisy_files[0], gt_files[0]))
    return pairs