from pathlib import Path
import torch

# The following is the hyper-params used in this project

# General Setting ---------------------------------------------------------------

WANDB_PROJECT = "Positive2Negative"
WANDB_FINETUNE_RUN = "SSL-ver1"

WEIGHTS_PATH = Path("unet_pretrain/unet_pretrain.pt")  # Adjust this according to the path
DATA_DIR     = Path("data")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_I = 1
MIN_I = 0

# -------------------------------------------------------------------------------

# Training Noise2NoiseUNet ------------------------------------------------------

CLEAN_DIR = Path("data/DIV2K")
OUTPUT_PATH = Path("unet_pretrain.pt")
CHECKPOINT_DIR = Path("unet_pretrain/checkpoints")

EPOCHS = 200
USE_PADDING = False # Toggle to switch between padding and cropping
BATCH_SIZE = 8
CROP_SIZE = 1024
NOISE_SIGMA = 25.0 # 25/255 -> 0.1
PRETRAIN_LR = 1e-3
WEIGHT_DECAY = 0.0

VAL_FRACTION = 0.05
NUM_WORKERS = 4
SEED = 42
SAVE_EVERY = 5
LOG_EVERY = 20

# -------------------------------------------------------------------------------

# Inferencing Noise2NoiseUNet ----------------------------------------------------

INFERENCE_RESULT_DIR = Path("result/inference_images")

# -------------------------------------------------------------------------------

# Finetuning Noise2NoiseUNet ----------------------------------------------------

FT_RESULT_DIR = Path("result/finetune_images")

NUM_ITERATION = 100
FT_CROP_SIZE = 2048

SIGMA = 0.75

LR = 1e-4
GAMMA_START = 2.0
GAMMA_END = 1.5
LOG_STEP = 1  # Print loss every N iterations.

# -------------------------------------------------------------------------------
