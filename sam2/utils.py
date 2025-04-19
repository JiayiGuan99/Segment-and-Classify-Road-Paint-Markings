import numpy as np
import torch
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from config import *

def build_model(checkpoint=SAM2_CHECKPOINT, cfg=MODEL_CFG, device="cuda"):
    """Build and return SAM2 model"""
    sam2_model = build_sam2(cfg, checkpoint, device=device)
    return SAM2ImagePredictor(sam2_model)

def get_points(mask, num_points):
    """Sample points inside the input mask"""
    points = []
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return np.array([])
        
    for i in range(num_points):
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return np.array(points)

def visualize_predictions(image, original_mask, predicted_mask):
    """Visualize the original image, ground truth mask, and predicted mask"""
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('Test Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Original Mask')
    plt.imshow(original_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Predicted Segmentation')
    plt.imshow(predicted_mask, cmap='jet')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_predictions(masks, scores):
    """Process predicted masks and create a segmentation map"""
    np_masks = np.array(masks[:, 0])
    np_scores = scores[:, 0]
    sorted_masks = np_masks[np.argsort(np_scores)][::-1]
    
    # Initialize segmentation map and occupancy mask
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
    
    # Combine masks to create the final segmentation map
    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        if mask.sum() == 0:
            continue
            
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue
        
        mask_bool = mask.astype(bool)
        mask_bool[occupancy_mask] = False  # Set overlapping areas to False in the mask
        seg_map[mask_bool] = i + 1  # Use boolean mask to index seg_map
        occupancy_mask[mask_bool] = True  # Update occupancy_mask
    
    return seg_map