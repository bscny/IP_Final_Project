# Positive2Negative: Breaking the Information-Lossy Barrier in Self-Supervised Single Image Denoising

**IMPORTANT**: This is the unofficial implementation of the CVPR 2025 accepted paper [Positive2Negative: Breaking the Information-Lossy Barrier in Self-Supervised Single Image Denoising](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Positive2Negative_Breaking_the_Information-Lossy_Barrier_in_Self-Supervised_Single_Image_Denoising_CVPR_2025_paper.pdf), we selected this paper as our 2026 spring Image Processing final project at Department of Computer science, NCCU

[[Final Presentation Slide](docs/Final_Slide)] [[Final Report](docs/Final_Reoort.pdf)]

**Note**: For the trained models in our experiment, Please see the below google drive link, we also provided the trained stats (e.g. loss curve)

## Table of content

- [Introduction](#introduction)
- [Take a Glance at the Result](#take-a-glance-at-the-result)
- [Trained Models and Stats](#trained-models-and-stats)
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

## Trained Models and Stats

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