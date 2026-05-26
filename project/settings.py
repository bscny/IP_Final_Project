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


# -------------------------------------------------------------------------------

# Finetuning Noise2NoiseUNet ----------------------------------------------------

NUM_ITERATION = 100

SIGMA = 0.75

LR = 1e-4
GAMMA_START = 2.0
GAMMA_END = 1.5
LOG_STEP = 1  # Print loss every N iterations.

# -------------------------------------------------------------------------------