# Positive2Negative: Breaking the Information-Lossy Barrier in Self-Supervised Single Image Denoising

**IMPORTANT**: This is the unofficial implementation of the CVPR 2025 accepted paper [Positive2Negative: Breaking the Information-Lossy Barrier in Self-Supervised Single Image Denoising](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Positive2Negative_Breaking_the_Information-Lossy_Barrier_in_Self-Supervised_Single_Image_Denoising_CVPR_2025_paper.pdf), we selected this paper as our 2026 spring Image Processing final project at Department of Computer science, NCCU

[[Final Presentation Slide](docs/Final_Slide.pdf) ([canva version](https://www.canva.com/design/DAHLDgymCfs/U95DF4bQAvnyfoeAaDFd0Q/edit))] [[Final Report](docs/Final_Reoort.pdf)]

**Note**: For the trained models in our experiment, Please see the below google drive link, we also provided the trained stats (e.g. loss curve)

## Table of content

- [Introduction](#introduction)
- [Take a Glance at the Result](#take-a-glance-at-the-result)
- [Instruction Steps](#instruction-steps)
- [File Structure](#file-structure)
- [Team Members](#team-members)

## Introduction

In this section, we introduce you what we have done on this project briefly:
1. We understood the core concept behind the paper and started implementing it via `PyTorch`
2. We gathered the required datasets and trained the base denoiser (Noise-to-Noise U-Net) using our own simple supervised method
3. We conducted the first experiment using only 1 model (the base denoiser) according to the paper, which is very ambiguously described, and got [miserable results](https://github.com/bscny/IP_Final_Project/tree/feat-finetune_and_inference/project/result/finetune_images)
4. After recieving advices from the professor, we switched to base + target (dual model) approach and recieve [better result](https://github.com/bscny/IP_Final_Project/tree/feat-finetune_and_inference/project/result/finetune_images_ver2) (but still a failure)
5. We analyzed the possible reasons behind the fail
6. We made the final slides and report to demonstrate our results, findings, and conclusions. The following section provides a quick view of it

To view our completely results:
1. switch to branch `feat-finetune_and_inference`
2. go to the `project/result` folder
3. `inference_images` contains the inference result from the baseline denoiser
4. `finetune_images` contains the only 1 model result of P2N training
5. `finetune_images_ver2` contains the base-target dual model approach's result
6. `finetune_images_ver3` also contains the base-target dual model approach's result but we try training from scratch with P2N
7. `finetune_images_ver4` is just testing, feel free to ignore

## Take a Glance at the Result

We first trained the base denoiser

![base loss](/docs/base_loss.png)

![base psnr](/docs/base_psnr.png)

After checking the base denoiser has the capabilities to denoise the image properly, we start training the target denoiser via P2N

![p2n loss](/docs/p2n_loss.png)

![p2n psnr](/docs/p2n_psnr.png)

We can see that the PSNR keeps going down despite the loss converges. After double checking with the visual evidence, we conclude that this training paradigm is defected

The Noisy Image:

![noise](/docs/noise.png)

Our baseline's Inference:

![baseline](/docs/baseline.png)

P2N's result:

![p2n](/docs/p2n.png)

Ground Truth:

![gt](/docs/gt.png)

## Instruction Steps

1. Follow `DATA_README.md` to gather all the datasets
2. Adjust the hyper-params in `settings.py`
3. Follow `/project/unet_pretrain/README.md` to get the base denoiser (Noise-2-Noise U-Net)
4. Simply `py unet_inference.py` to get the inference result of the baseline model
5. Adjust the hyper-params in `settings.py`
6. Simply `py finetune.py` to get the run P2N training loop on a single image
7. View the result in `result` folder

## File Structure

```
project/
├── data
│   ├── CC
│   ├── DIV2K
│   ├── PolyU
│   └── SIDD
├── result
├── src
│   ├── p2n.py
│   └── unet.py
├── README.md
├── settings.py
├── train.py
└── finetune.py
```

## Team Members

- 游宗諺
- 王冠智