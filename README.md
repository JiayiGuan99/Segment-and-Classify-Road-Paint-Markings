# Pavement Segmentation using SAM2

This repository contains code for fine-tuning the Segment Anything 2 (SAM2) model for pavement segmentation in satellite imagery. The project is structured to allow for easy configuration, training, and testing of the model.

## Project Structure

```
.
├── config.py          # Configuration parameters
├── dataloader.py      # Data loading and processing utilities
├── train.py           # Training script
├── utils.py           # Utility functions
├── requirements.txt   # Project dependencies
└── checkpoints/       # Directory for model checkpoints
```

## Installation

### 1. Clone the SAM2 Repository

First, clone the SAM2 repository from Facebook Research:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

### 2. Clone This Repository/move all the files in to sam2 folder



### 3. Install Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Before running the training script, you'll need to modify the parameters in `config.py` to match your setup:

```python
# Dataset paths
DATA_DIR = "path/to/your/dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "sat_pave_dataset/selection_org")
MASKS_DIR = os.path.join(DATA_DIR, "sat_pave_dataset/selection_label")
TRAIN_CSV = os.path.join(DATA_DIR, "sat_pave_dataset/train.csv")

# Model paths
SAM2_CHECKPOINT = "path/to/sam2.1_hiera_small.pt"
MODEL_CFG = "path/to/sam2.1_hiera_s.yaml"
```

You'll need to download the SAM2 checkpoint file from the [SAM2 repository](https://github.com/facebookresearch/sam2) or use the following:

- `sam2.1_hiera_tiny.pt`
- `sam2.1_hiera_small.pt`
- `sam2.1_hiera_base_plus.pt`
- `sam2.1_hiera_large.pt`

Along with the corresponding config files:

- `sam2.1_hiera_t.yaml`
- `sam2.1_hiera_s.yaml`
- `sam2.1_hiera_b+.yaml`
- `sam2.1_hiera_l.yaml`

### Training Parameters

You can also adjust the training parameters in `config.py`:

```python
NO_OF_STEPS = 3000
ACCUMULATION_STEPS = 4
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
STEP_SIZE = 500
LR_GAMMA = 0.2
```

## Dataset Format

The dataset should be organized with:

1. Images in the `IMAGES_DIR` directory
2. Masks in the `MASKS_DIR` directory
3. A CSV file at `TRAIN_CSV` with columns:
   - `ImageId`: The filename of the image
   - `MaskId`: The filename of the corresponding mask

## Training

To train the model, run:

```bash
python train.py
```

This will:
1. Load the dataset and split it into training and testing sets
2. Build the SAM2 model
3. Train the model for the specified number of steps
4. Save checkpoints in the `checkpoints` directory
5. Save the best model based on IoU performance
6. Test the model on a random image from the test set

## Testing

To test the model on a specific checkpoint, modify the `test_model` function in `train.py`:

```python
test_model(predictor=None, test_data=test_data, model_weights="checkpoints/fine_tuned_sam2_best.torch")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project builds upon the Segment Anything 2 (SAM2) model from Facebook Research. For more information about SAM2, please visit the [official repository](https://github.com/facebookresearch/sam2).

## Citation

```
@inproceedings{ravi2024sam2,
  title={Segment Anything 2},
  author={Ravi, Viraj and Xia, Enze and Wang, Jeremy and Ryali, Chaitanya and Ghiasi, Golnaz and Dehghani, Mostafa and Lu, Ekin and Kamath, Priya and Ramanan, Deva and Kirillov, Alexander},
  booktitle={ECCV},
  year={2024}
}
```

And the DataCamp tutorial:

```
DataCamp. (2024). Fine-tuning SAM2 for Medical Image Segmentation. 
Retrieved from https://www.datacamp.com/tutorial/sam2-fine-tuning
```