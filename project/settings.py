from pathlib import Path
import torch

# The following is the hyper-params used in this project

# General Setting ---------------------------------------------------------------

WEIGHTS_PATH = Path("pretrained_weights.pt")  # Adjust this according to the path
DATA_DIR     = Path("data")
RESULT_DIR   = Path("result")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_I = 1
MIN_I = 0

# -------------------------------------------------------------------------------

# Training Noise2NoiseUNet ------------------------------------------------------

CLEAN_DIR = Path("div2k/div2k_train")
OUTPUT_PATH = Path("unet_pretrain.pt")
CHECKPOINT_DIR = Path("unet_pretrain/checkpoints")

EPOCHS = 100
BATCH_SIZE = 8
CROP_SIZE = 256
NOISE_SIGMA = 25.0 # 25/255 -> 0.1
PRETRAIN_LR = 1e-4
WEIGHT_DECAY = 0.0

VAL_FRACTION = 0.05
NUM_WORKERS = 4
SEED = 42
SAVE_EVERY = 5
LOG_EVERY = 20

# -------------------------------------------------------------------------------

# Finetuning Noise2NoiseUNet ----------------------------------------------------

NUM_ITERATION = 100

SIGMA = 0.75

LR = 1e-4
GAMMA_START = 2.0
GAMMA_END = 1.5
LOG_STEP = 1  # Print loss every N iterations.

# -------------------------------------------------------------------------------
