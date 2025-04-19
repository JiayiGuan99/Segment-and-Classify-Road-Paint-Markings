import os

# Dataset paths
DATA_DIR = "D:/VCLab2/final_project/dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "sat_pave_dataset/selection_org")
MASKS_DIR = os.path.join(DATA_DIR, "sat_pave_dataset/selection_label")
TRAIN_CSV = os.path.join(DATA_DIR, "sat_pave_dataset/train.csv")

# Model paths
SAM2_CHECKPOINT = "D:/VCLab2/final_project/dataset/sat_pave_dataset/sam2.1_hiera_small.pt"
MODEL_CFG = "D:/VCLab2/final_project/dataset/sat_pave_dataset/sam2.1_hiera_s.yaml"

# Checkpoint directory
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Create the directory if it doesn't exist

# Training parameters
NO_OF_STEPS = 5000
ACCUMULATION_STEPS = 4
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
STEP_SIZE = 500
LR_GAMMA = 0.2
FINE_TUNED_MODEL_NAME = "fine_tuned_sam2"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Image parameters
MAX_SIZE = 1024