import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def segment_image(
    image_path,
    model_cfg_path,
    checkpoint_path,
    fine_tuned_weights_path,
    num_points=30,
    output_dir="output",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Segment an image using a fine-tuned SAM2 model.
    
    Args:
        image_path: Path to the input image
        model_cfg_path: Path to the SAM2 model configuration file
        checkpoint_path: Path to the SAM2 base checkpoint
        fine_tuned_weights_path: Path to the fine-tuned weights
        num_points: Number of random points to generate for prompting
        output_dir: Directory to save the output masks
        device: Device to run inference on ("cuda" or "cpu")
    
    Returns:
        segmentation_map: The final segmentation map
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB
    image = image[..., ::-1]
    
    # Resize image to fit within 1024x1024 while preserving aspect ratio
    r = min(1024 / image.shape[1], 1024 / image.shape[0])
    resized_image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))
    
    # Save original size for later
    original_h, original_w = image.shape[:2]
    
    # Build SAM2 model
    print("Loading model...")
    sam2_model = build_sam2(model_cfg_path, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Load fine-tuned weights
    print(f"Loading fine-tuned weights from {fine_tuned_weights_path}")
    predictor.model.load_state_dict(torch.load(fine_tuned_weights_path, map_location=device))
    predictor.model.eval()
    
    # Set the image
    print("Processing image...")
    predictor.set_image(resized_image)
    
    # Generate random points for prompting
    # For a single image with no mask, use simple grid sampling
    h, w = resized_image.shape[:2]
    grid_points = []
    
    # Create a grid of points
    rows, cols = 5, 6  # Adjust grid density as needed
    for i in range(rows):
        for j in range(cols):
            x = int(w * (j + 0.5) / cols)
            y = int(h * (i + 0.5) / rows)
            grid_points.append([[x, y]])
    
    # Convert to numpy array
    input_points = np.array(grid_points)
    
    # Run inference
    print(f"Running inference with {len(input_points)} prompt points...")
    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )
    
    # Process the predicted masks
    np_masks = masks[:, 0]  # Get the first mask for each point
    np_scores = scores[:, 0]
    
    # Sort masks by score (highest first)
    sorted_indices = np.argsort(np_scores)[::-1]
    sorted_masks = np_masks[sorted_indices]
    sorted_scores = np_scores[sorted_indices]
    
    # Initialize segmentation map and occupancy mask
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
    
    # Combine masks to create the final segmentation map
    print("Creating final segmentation map...")
    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        if mask.sum() == 0:  # Skip empty masks
            continue
            
        # Skip if too much overlap with existing mask
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue
        
        mask_bool = mask.astype(bool)
        mask_bool[occupancy_mask] = False  # Remove overlapping regions
        seg_map[mask_bool] = i + 1  # Assign unique label
        occupancy_mask[mask_bool] = True
    
    # Resize segmentation map back to original image size
    seg_map_original = cv2.resize(
        seg_map, 
        (original_w, original_h), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Create binary mask (0 for background, 1 for any segment)
    binary_mask = (seg_map > 0).astype(np.uint8) * 255
    binary_mask_original = cv2.resize(
        binary_mask, 
        (original_w, original_h), 
        interpolation=cv2.INTER_NEAREST
    )
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save binary mask
    cv2.imwrite(
        os.path.join(output_dir, f"{base_name}_binary_mask.png"), 
        binary_mask_original
    )
    
    # Save colored segmentation map
    plt.figure(figsize=(12, 12))
    plt.imshow(seg_map, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_segmentation.png"))
    plt.close()
    
    # Save visualization (side by side)
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(resized_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Binary Mask')
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Segmentation Map')
    plt.imshow(seg_map, cmap='jet')
    for i, point in enumerate(input_points):
        plt.scatter(point[0][0], point[0][1], c='white', s=20, alpha=0.7)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_visualization.png"))
    plt.close()
    
    print(f"Results saved to {output_dir}")
    return seg_map, binary_mask

if __name__ == "__main__":
    # Configuration - update these paths according to your setup
    IMAGE_PATH = "C:/Users/Shikaka/Desktop/Image_20250404135013.png"
    MODEL_CFG_PATH = "D:/VCLab2/final_project/dataset/sat_pave_dataset/sam2.1_hiera_s.yaml"
    CHECKPOINT_PATH = "D:/VCLab2/final_project/dataset/sat_pave_dataset/sam2.1_hiera_small.pt"
    FINE_TUNED_WEIGHTS_PATH = "checkpoints/fine_tuned_sam2_weighted_loss_final.torch"
    
    # Run segmentation
    seg_map, binary_mask = segment_image(
        image_path=IMAGE_PATH,
        model_cfg_path=MODEL_CFG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        fine_tuned_weights_path=FINE_TUNED_WEIGHTS_PATH,
        num_points=30,
        output_dir="output"
    )
    
    # Display results (if running in interactive environment)
    try:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Binary Mask")
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Segmentation Map")
        plt.imshow(seg_map, cmap='jet')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    except:
        pass
    
    print("Segmentation complete!")