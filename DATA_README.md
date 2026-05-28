# Dataset Used in this Project

We have used [DIV2K](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images?select=DIV2K_train_HR) for training, and [SIDD](https://www.kaggle.com/datasets/rajat95gupta/smartphone-image-denoising-dataset), [CC (subset)](https://github.com/csjunxu/MCWNNM-ICCV2017/tree/master), [PolyU](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset).

## Training U-Net

## Finetuning the U-Net via Self-Supervised Learning

Before start downloading the datasets, first `cd` to the project root and make sure there are a `data/` folder there.

### SIDD

1. `curl -Lo ./SIDD.zip  https://www.kaggle.com/api/v1/datasets/download/rajat95gupta/smartphone-image-denoising-dataset`
2. `unzip SIDD.zip`
3. `rm SIDD.zip`
4. `mv SIDD_Small_sRGB_Only/Data/ data/SIDD`
5. `rm -rf SIDD_Small_sRGB_Only/`

### CC (15 image subset)

1. `git clone https://github.com/csjunxu/MCWNNM-ICCV2017.git`
2. `mv MCWNNM-ICCV2017/Real_ccnoise_denoised_part/ data/CC`
3. `rm data/CC/*ours.png`
4. `rm -rf MCWNNM-ICCV2017/`

### PolyU

1. `git clone https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset.git`
2. `mv PolyU-Real-World-Noisy-Images-Dataset/OriginalImages/ data/PolyU`
3. `rm -rf PolyU-Real-World-Noisy-Images-Dataset/`

### div2k(only train is used)

1. `cd project/data/div2k`
2. `uv run project/data/div2k/download.py` 
3. `mv ./data/div2k/datasets/soumikrakshit/div2k-high-resolution-images/versions/1 .`
4. `mv 1/DIV2K_train_HR/ div2k_train`
5. `rm -rf data/ 1 `

## Final Look

After the above steps, the file structure under `data` is:
```
data
├── CC
│   ├── ....
│   ├── d800_iso6400_3_mean.png
│   └── d800_iso6400_3_real.png
├── PolyU
│   ├── ...
│   ├── SonyA7II_water_mean.JPG
│   └── SonyA7II_water_Real.JPG
└── SIDD
    ├── ...
    ├── 0199_010_GP_00800_01600_5500_N
    │   ├── GT_SRGB_010.PNG
    │   └── NOISY_SRGB_010.PNG
    └── 0200_010_GP_01600_03200_5500_N
        ├── GT_SRGB_010.PNG
        └── NOISY_SRGB_010.PNG
```
