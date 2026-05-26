# Image Processing Final Project Proposal

Member: 111703040 游宗諺, 115753205 王冠智

## [Positive2Negative: Breaking the Information-Lossy Barrier in Self-Supervised Single Image Denoising](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Positive2Negative_Breaking_the_Information-Lossy_Barrier_in_Self-Supervised_Single_Image_Denoising_CVPR_2025_paper.pdf)

Conference: CVPR
Year: 2025
Category: Image Restoration (Denoising) (Self-Supervised Learning)

## Introduction and Problem Description

Current research regarding image denoising has achieved remarkable results using supervised methods trained on paired clean-noisy image datasets. However, supervised methods fall short in dynamic scenes where datasets are small and pure clean images are typically difficult to acquire, making supervised learning impossible.

To address this, existing methods utilize self-supervised learning to denoise a given image. Currently, these self-supervised image denoising methods primarily fall into two categories:
- **Noise2Noise-Based Paradigms**: Attempt to construct independent noisy observations from a single image by adding additional noise or downsampling the image.
- **Noise2Void-Based Paradigms**: utilize masking strategies on input image to predict a masked central pixel using only surrounding data.

Consequently, these 2 paradigms suffer from "information-lossy" issue because a part of the original dirty image is destroyed because of the added noise or cropped pixels. This barrier inevitably compromises the model's denoising capabilities, meaning the final generated images often suffer from residual noise, aliasing effects, or noticeable texture loss.

This paper presents a new paradigm that breaks this "information-lossy" barrier. The framework achieves lossless training data generation through two main steps:
1. **Renoised Data Construction (RDC)**: leveraging a pre-trained U-Net, RDC extracts the predicted noise from a forward pass and uses a zero-mean sampling strategy to generate "Positive" and "Negative" noises. These are added back to the predicted denoised image to construct a dynamic pair of noisy training images.
2. **Denoised Consistency Supervision (DCS)**: Given the newly constructed noisy images (a pair), DCS acts as a supervision strategy. Specifically, we try to fine-tune the pre-trained U-Net by minimizing the difference between the predicted denoised outputs of these two newly constructed images.

## Project Goal and Implementation Steps

The objective of this project is to reproduce and evaluate the Positive2Negative (P2N) paradigm, specifically, the project will go through the following steps:
1. Implement the U-Net from an 2018 ICML paper ([Noise2Noise](https://arxiv.org/pdf/1803.04189)'s Appendix Table 2)
2. Get the [DIV2K](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images?select=DIV2K_train_HR) and apply Gaussian noise using OpenCV
3. Train the U-Net with **Supervised Learning** method (using MSE as loss, the ground truth is the clean img)
4. Now we have the pre-trained U-Net (a denoiser), we start implementing paper's method by first preparing **ONLY 1** dirty image, say `Y`
5. run `Y` on inference mode to get an initially poorly predicted denoised image, say `X^head`, and get the predicted noise by "`Y` - `X^head`"
6. Add the predicted noise back to the `X^head` with 2 opposite scalers (this scalers follows _Normal(1, 0.75)_ ), now we have 2 images, say `Y_p` and `Y_n`
7. run `Y_p` and `Y_n` on training mode to compute the loss function and do gradient descent (This is the **Self-Supervised Learning** part)
8. repeat step `5` ~ `7` for 100 iterations
9. Finally, run `Y` on inference mode to get the "fine-tuned" denoised image

The `Y` will be all the image in [SIDD](https://www.kaggle.com/datasets/rajat95gupta/smartphone-image-denoising-dataset), [CC (subset)](https://github.com/csjunxu/MCWNNM-ICCV2017/tree/master), and [PolyU](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset) evaluation datasets (using PSNR benchmark).

## Prerequisite Knowledge

To successfully complete this project, the following knowledge areas are required:
1. Basic CNN knowledge (U-Net, skip connections)
2. PyTorch Programming
3. Image Processing Fundamentals (add noise and random crop)
4. Evaluation Metrics (PSNR)
5. supervised learning vs self-supervised learning