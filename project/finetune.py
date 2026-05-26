import torch

# Custom Modules
from src.unet import Noise2NoiseUNet
from src.p2n import train_p2n
from src.utils.image_helper import compute_psnr, load_image_tensor, save_tensor_image, pad_to_multiple, unpad
from src.utils.data_helper import find_cc_pairs, find_polyu_pairs, find_sidd_pairs

import settings

def main():       
    # Ensure pre-trained weights exist
    if not settings.WEIGHTS_PATH.exists():
        print(f"ERROR: Pre-trained weights not found at '{settings.WEIGHTS_PATH}'")
        return

    pretrained_state = torch.load(settings.WEIGHTS_PATH, map_location="cpu")

    dataset_configs = [
        ("CC",    find_cc_pairs(settings.DATA_DIR / "CC")),
        ("PolyU", find_polyu_pairs(settings.DATA_DIR / "PolyU")),
        ("SIDD",  find_sidd_pairs(settings.DATA_DIR / "SIDD")),
    ]

    summary = {}   # Map for dataset_name -> (list_of_psnrs, average_psnr)

    for ds_name, pairs in dataset_configs:
        if not pairs:
            print(f"[{ds_name}] No pairs found — skipping.\n")
            continue

        print(f"{'='*60}")
        print(f" Dataset: {ds_name}  ({len(pairs)} image pairs)")
        print(f"{'='*60}")

        psnr_list = []

        for idx, (noisy_path, gt_path) in enumerate(pairs, 1):
            print(f"\n[{ds_name}] ({idx}/{len(pairs)}) {noisy_path.name}")

            # Load Images
            noisy = load_image_tensor(noisy_path, settings.DEVICE, settings.MIN_I, settings.MAX_I)
            gt    = load_image_tensor(gt_path,    settings.DEVICE, settings.MIN_I, settings.MAX_I)

            # Pad for UNet (Height and Width must be divisible by 32)
            noisy_pad, pad_hw = pad_to_multiple(noisy, 32)

            # Initialize Model (Fresh for each image)
            model = Noise2NoiseUNet().to(settings.DEVICE)
            model.load_state_dict(pretrained_state, strict=True)

            # Fine-Tune using P2N
            denoised_pad = train_p2n(
                model=model,
                dirty_img=noisy_pad,
                num_iterations=settings.NUM_ITERATION,
                lr=settings.LR,
                sigma=settings.SIGMA,
                gamma_start=settings.GAMMA_START,
                gamma_end=settings.GAMMA_END,
                log_every=settings.LOG_STEP,
                min_i=settings.MIN_I,
                max_i=settings.MAX_I
            )

            # Restore original dimensions
            denoised = unpad(denoised_pad, pad_hw)

            # Metrics & Saving
            psnr = compute_psnr(denoised, gt, settings.MIN_I, settings.MAX_I)
            psnr_list.append(psnr)
            print(f"  PSNR: {psnr:.4f} dB")

            out_path = settings.RESULT_DIR / ds_name / noisy_path.name
            save_tensor_image(denoised, out_path, settings.MIN_I, settings.MAX_I)
            print(f"  Saved → {out_path}")

            # Clean up memory dynamically 
            del model
            del denoised_pad
            torch.cuda.empty_cache()

        avg = sum(psnr_list) / len(psnr_list)
        summary[ds_name] = (psnr_list, avg)
        print(f"\n[{ds_name}] Average PSNR: {avg:.4f} dB\n")

    # Final Benchmarking Summary
    print(f"\n{'='*60}")
    print(" SUMMARY")
    print(f"{'='*60}")
    for ds_name, (psnr_list, avg) in summary.items():
        print(f"  {ds_name:<8}  avg PSNR = {avg:.4f} dB  (over {len(psnr_list)} images)")
    print()

if __name__ == "__main__":
    main()