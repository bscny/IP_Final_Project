import torch
import re
import wandb

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
    
    wandb.init(
        project=settings.FT_WANDB_PROJECT,
        name=settings.FT_WANDB_RUN,
        config={
            "learning_rate": settings.LR,
            "iterations": settings.NUM_ITERATION,
            "sigma": settings.SIGMA,
            "gamma_start": settings.GAMMA_START,
            "gamma_end": settings.GAMMA_END
        }
    )

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

        psnr_list = []          # The evaluation PSNR result for each image
        ds_loss_histories = []  # For wandb, track each iteration for each image
        ds_psnr_histories = []  # For wandb, track each iteration for each image
        recorded_steps = []     # For wandb, track each iteration for each image

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
            denoised_pad, steps, losses, psnrs = train_p2n(
                model=model,
                dirty_img=noisy_pad,
                gt_img=gt,
                pad_hw=pad_hw,
                num_iterations=settings.NUM_ITERATION,
                lr=settings.LR,
                sigma=settings.SIGMA,
                gamma_start=settings.GAMMA_START,
                gamma_end=settings.GAMMA_END,
                log_every=settings.LOG_STEP,
                min_i=settings.MIN_I,
                max_i=settings.MAX_I,
                crop_size=settings.FT_CROP_SIZE
            )

            # Save the data points for this image
            ds_loss_histories.append(losses)
            ds_psnr_histories.append(psnrs)
            if not recorded_steps:
                recorded_steps = steps

            # Restore original dimensions
            denoised = unpad(denoised_pad, pad_hw)

            # Metrics & Saving
            psnr = compute_psnr(denoised, gt, settings.MIN_I, settings.MAX_I)
            psnr_list.append(psnr)
            print(f"  PSNR: {psnr:.4f} dB")

            # Clean the stem (removes '_real' or '_Real' for CC and PolyU)
            clean_stem = re.sub(r"_real$", "", noisy_path.stem, flags=re.IGNORECASE)
            
            # Clean SIDD prefix and PREVENT OVERWRITES
            if ds_name == "SIDD":
                clean_stem = re.sub(r"^NOISY_", "", clean_stem, flags=re.IGNORECASE)
                # Attach the parent directory name (e.g., '0199_010_GP_00800...') to the stem
                clean_stem = f"{noisy_path.parent.name}_{clean_stem}"

            # Construct the new filename appending '_ours' and keeping the original extension
            our_name = f"{clean_stem}_ours{noisy_path.suffix}"
            
            # Construct the new output path
            out_path = settings.FT_RESULT_DIR / ds_name / our_name
            
            # Ensure the new directory structure exists before saving
            out_path.parent.mkdir(parents=True, exist_ok=True)

            save_tensor_image(denoised, out_path, settings.MIN_I, settings.MAX_I)
            print(f"  Saved → {out_path}")

            # Clean up memory dynamically 
            del model
            del denoised_pad
            torch.cuda.empty_cache()

        # Calculate and log mean loss curve for the dataset
        # zip(*ds_loss_histories) groups the losses by iteration across all images
        mean_losses = [sum(iter_losses) / len(iter_losses) for iter_losses in zip(*ds_loss_histories)]
        mean_psnrs  = [sum(iter_psnrs) / len(iter_psnrs) for iter_psnrs in zip(*ds_psnr_histories)]

        # Graph to wandb
        for step, mean_loss, mean_psnr in zip(recorded_steps, mean_losses, mean_psnrs):
            wandb.log({
                f"{ds_name}/iteration": step,
                f"{ds_name}/mean_loss": mean_loss,
                f"{ds_name}/mean_psnr": mean_psnr
            })

        avg_eval_psnr = sum(psnr_list) / len(psnr_list)
        summary[ds_name] = (psnr_list, avg_eval_psnr)
        wandb.log({f"{ds_name}/avg_final_psnr": avg_eval_psnr})
        # print(f"\n[{ds_name}] Average PSNR: {avg_eval_psnr:.4f} dB\n")

    # Final Benchmarking Summary
    print(f"\n{'='*60}")
    print(" SUMMARY")
    print(f"{'='*60}")
    for ds_name, (psnr_list, avg_eval_psnr) in summary.items():
        print(f"  {ds_name:<8}  avg_eval_psnr PSNR = {avg_eval_psnr:.4f} dB  (over {len(psnr_list)} images)")
    print()

if __name__ == "__main__":
    main()