# Full script for SAM2 fine-tuning with manual correction capabilities
import os
import random
import pandas as pd
import cv2
import torch
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
from sklearn.model_selection import train_test_split
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Path configurations
data_dir = "D:/VCLab2/final_project/dataset"
images_dir = os.path.join(data_dir, "sat_pave_dataset/selection_org")
masks_dir = os.path.join(data_dir, "sat_pave_dataset/selection_label")

# Model configurations
sam2_checkpoint = "D:/VCLab2/final_project/dataset/sat_pave_dataset/sam2.1_hiera_small.pt"
model_cfg = "D:/VCLab2/final_project/dataset/sat_pave_dataset/sam2.1_hiera_s.yaml"
FINE_TUNED_MODEL_WEIGHTS = "checkpoints/fine_tuned_sam2_weighted_loss_best.torch"

# Load CSV file with image and mask mappings
train_df = pd.read_csv(os.path.join(data_dir, "sat_pave_dataset/train.csv"))

# Split into train and test sets
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Prepare training and testing data lists
train_data = []
for index, row in train_df.iterrows():
    train_data.append({
        "image": os.path.join(images_dir, row['ImageId']),
        "annotation": os.path.join(masks_dir, row['MaskId'])
    })

test_data = []
for index, row in test_df.iterrows():
    test_data.append({
        "image": os.path.join(images_dir, row['ImageId']),
        "annotation": os.path.join(masks_dir, row['MaskId'])
    })

def read_image(image_path, mask_path):
    """Read and resize image and mask"""
    img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
    mask = cv2.imread(mask_path, 0)
    
    # Handle potential file reading errors
    if img is None or mask is None:
        raise ValueError(f"Could not read image or mask from {image_path} or {mask_path}")
    
    # Resize to fit within 1024x1024
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    return img, mask

def get_points(mask, num_points):
    """Sample random points inside the mask"""
    points = []
    coords = np.argwhere(mask > 0)
    
    # Handle empty masks
    if len(coords) == 0:
        return np.array([])
    
    for i in range(num_points):
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])  # x, y format
    return np.array(points)

def add_manual_correction_points(predictor, image, existing_points=None, existing_labels=None):
    """
    Interactive function to manually add correction points to SAM2 segmentation.
    
    Args:
        predictor: SAM2ImagePredictor instance with image already set
        image: Original image for display
        existing_points: Optional array of existing points
        existing_labels: Optional array of existing labels (1 for foreground, 0 for background)
        
    Returns:
        Tuple of (points_array, labels_array) for prediction
    """
    # Initialize points and labels if not provided
    if existing_points is None:
        points = []
    else:
        points = existing_points.squeeze(1).tolist() if len(existing_points.shape) > 2 else existing_points.tolist()
    
    if existing_labels is None:
        labels = []
    else:
        # Ensure labels are simple integers, not nested arrays
        try:
            labels = existing_labels.flatten().tolist()
        except:
            # If flattening fails, handle as simple list
            labels = [int(l) for l in existing_labels.reshape(-1)]
    
    # Create figure and axes for interaction
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title('Click to add points: Left-click for foreground (green), Right-click for background (red)')
    
    # Plot existing points if any
    for i, (point, label) in enumerate(zip(points, labels)):
        color = 'green' if label == 1 else 'red'
        ax.plot(point[0], point[1], 'o', color=color, markersize=8)
    
    # Track point markers for undo functionality
    point_markers = []
    
    def update_preview():
        """Update segmentation preview using current points"""
        if len(points) == 0:
            print("No points added yet. Add at least one point to preview.")
            return

        try:
            points_array = np.array(points).reshape(-1, 1, 2)
            labels_array = np.array([int(l) for l in labels]).reshape(-1, 1)
            
            # Predict masks with current points
            with torch.no_grad():
                masks, scores, _ = predictor.predict(
                    point_coords=points_array,
                    point_labels=labels_array
                )
            
            # Display result in separate figure
            preview_fig, preview_ax = plt.subplots(1, 2, figsize=(12, 6))
            preview_ax[0].imshow(image)
            preview_ax[0].set_title('Original Image with Points')
            
            # Plot points on preview
            for i, (point, label) in enumerate(zip(points, labels)):
                color = 'green' if label == 1 else 'red'
                preview_ax[0].plot(point[0], point[1], 'o', color=color, markersize=8)
            
            # Show mask prediction
            preview_ax[1].imshow(image)
            preview_ax[1].imshow(masks[0, 0], alpha=0.5, cmap='jet')
            preview_ax[1].set_title(f'Predicted Mask (Score: {scores[0, 0]:.4f})')
            
            preview_fig.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in preview generation: {e}")
            print("Points:", points)
            print("Labels:", labels)
    
    # Define click event handler
    def onclick(event):
        if event.inaxes != ax:
            return
        
        if event.button == 1:  # Left click (foreground)
            point = [event.xdata, event.ydata]
            points.append(point)
            labels.append(1)
            marker = ax.plot(event.xdata, event.ydata, 'go', markersize=8)[0]
            point_markers.append(marker)
            plt.draw()
            print(f"Added positive point at ({event.xdata:.1f}, {event.ydata:.1f})")
        elif event.button == 3:  # Right click (background)
            point = [event.xdata, event.ydata]
            points.append(point)
            labels.append(0)
            marker = ax.plot(event.xdata, event.ydata, 'ro', markersize=8)[0]
            point_markers.append(marker)
            plt.draw()
            print(f"Added negative point at ({event.xdata:.1f}, {event.ydata:.1f})")
    
    # Set up button for previewing results
    preview_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
    preview_button = Button(preview_ax, 'Preview')
    preview_button.on_clicked(lambda event: update_preview())
    
    # Set up button for completing the process
    done_ax = plt.axes([0.81, 0.05, 0.1, 0.075])
    done_button = Button(done_ax, 'Done')
    
    # Add undo button
    undo_ax = plt.axes([0.59, 0.05, 0.1, 0.075])
    undo_button = Button(undo_ax, 'Undo')
    
    def undo(event):
        if len(points) > 0:
            points.pop()
            labels.pop()
            marker = point_markers.pop()
            marker.remove()
            plt.draw()
            print("Undid last point")
    
    undo_button.on_clicked(undo)
    
    # Connect the click event
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    # Blocking show to wait for user interaction
    done_pressed = [False]
    def on_done(event):
        plt.close(fig)
        done_pressed[0] = True
    
    done_button.on_clicked(on_done)
    plt.show()
    
    # Convert lists to numpy arrays in the format expected by the predictor
    try:
        # Convert lists to numpy arrays in the format expected by the predictor
        points_array = np.array(points).reshape(-1, 1, 2)
        labels_array = np.array([int(l) for l in labels]).reshape(-1, 1)  # Force conversion to int
    except ValueError as e:
        print(f"Warning: {e}")
        print("Converting labels to proper format...")
        # Emergency fix - ensure labels are integers
        labels = [int(l) if isinstance(l, (int, float)) else 1 for l in labels]
        points_array = np.array(points).reshape(-1, 1, 2)
        labels_array = np.array(labels).reshape(-1, 1)
    
    print(f"Final points count: {len(points)} (Positive: {labels.count(1)}, Negative: {labels.count(0)})")
    return points_array, labels_array

def run_segmentation_with_manual_correction():
    """Main function to run segmentation with manual correction"""
    # Load the model
    print("Loading SAM2 model...")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    
    # Build predictor and load weights
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Check if fine-tuned weights exist and load them
    if os.path.exists(FINE_TUNED_MODEL_WEIGHTS):
        print(f"Loading fine-tuned weights: {FINE_TUNED_MODEL_WEIGHTS}")
        predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))
    else:
        print("Fine-tuned weights not found. Using base model.")
    
    # Select a test image randomly
    selected_entry = random.choice(test_data)
    image_path = selected_entry['image']
    mask_path = selected_entry['annotation']
    
    print(f"Selected image: {os.path.basename(image_path)}")
    print(f"Selected mask: {os.path.basename(mask_path)}")
    
    # Always load fresh ground truth mask directly from file
    image, original_gt_mask = read_image(image_path, mask_path)
    
    # Make a copy of the original ground truth to ensure it remains unchanged
    ground_truth_mask = original_gt_mask.copy()
    
    # Set the image in the predictor
    predictor.set_image(image)
    
    # Generate initial automatic points
    num_auto_points = 5
    auto_points = get_points(ground_truth_mask, num_auto_points)
    
    # If no points were found (empty mask), create a default point
    if len(auto_points) == 0:
        print("Warning: Empty mask detected. Creating default point.")
        h, w = image.shape[:2]
        auto_points = np.array([[[w//2, h//2]]])  # Center point
    
    auto_labels = np.ones([auto_points.shape[0], 1])
    
    # Initial prediction with automatic points
    print(f"Making initial prediction with {len(auto_points)} automatic points...")
    with torch.no_grad():
        masks, scores, _ = predictor.predict(
            point_coords=auto_points,
            point_labels=auto_labels
        )
    
    # Display initial results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(ground_truth_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f'Initial Prediction (Score: {scores[0, 0]:.4f})')
    plt.imshow(image)
    plt.imshow(masks[0, 0], alpha=0.5, cmap='jet')
    for point in auto_points:
        plt.plot(point[0][0], point[0][1], 'go', markersize=8)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Ask if manual correction is needed
    correction_needed = input("Do you want to manually correct the segmentation? (y/n): ").lower().strip() == 'y'
    
    final_masks = masks  # Default to initial prediction
    
    if correction_needed:
        print("\nManual correction mode:")
        print("- Left-click to add positive points (areas that SHOULD be segmented)")
        print("- Right-click to add negative points (areas that should NOT be segmented)")
        print("- Click 'Preview' to see the current segmentation")
        print("- Click 'Undo' to remove the last point")
        print("- Click 'Done' when finished\n")
        
        # Get manual correction points
        try:
            manual_points, manual_labels = add_manual_correction_points(
                predictor, 
                image,
                existing_points=auto_points,
                existing_labels=auto_labels
            )
            
            # Make prediction with manual points
            print("Generating final prediction with manual correction points...")
            with torch.no_grad():
                final_masks, scores, logits = predictor.predict(
                    point_coords=manual_points,
                    point_labels=manual_labels
                )
        except Exception as e:
            print(f"Error during manual correction: {e}")
            print("Continuing with initial prediction...")
    else:
        print("Using initial prediction without manual corrections.")
    
    # Process masks
    np_masks = np.array(final_masks[:, 0])
    np_scores = scores[:, 0]
    sorted_indices = np.argsort(np_scores)[::-1]
    sorted_masks = np_masks[sorted_indices]
    sorted_scores = np_scores[sorted_indices]
    
    # Create final segmentation map
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
    
    # Combine masks to create the final segmentation map
    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        # Skip if there's significant overlap with existing segments
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue
        
        mask_bool = mask.astype(bool)
        mask_bool[occupancy_mask] = False  # Set overlapping areas to False
        seg_map[mask_bool] = i + 1  # Assign segment ID
        occupancy_mask[mask_bool] = True  # Update occupancy mask
    
    # Final visualization
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Ground Truth Mask')
    # Always use the original ground truth mask
    plt.imshow(ground_truth_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Final Segmentation')
    plt.imshow(seg_map, cmap='jet')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Compare with ground truth - use the original ground truth mask
    binary_gt = (ground_truth_mask > 0).astype(np.uint8)
    binary_pred = (seg_map > 0).astype(np.uint8)
    
    # Calculate IoU
    intersection = (binary_gt & binary_pred).sum()
    union = (binary_gt | binary_pred).sum()
    iou = intersection / union if union > 0 else 0
    
    print(f"Final segmentation IoU: {iou:.4f}")
    
    # Save results if needed
    save_results = input("Do you want to save the results? (y/n): ").lower().strip() == 'y'
    if save_results:
        results_dir = "segmentation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save original image
        plt.imsave(os.path.join(results_dir, f"{base_name}_image.png"), image)
        
        # Save ground truth mask - ensure we save the original
        plt.imsave(os.path.join(results_dir, f"{base_name}_gt_mask.png"), ground_truth_mask, cmap='gray')
        
        # Save prediction
        plt.figure(figsize=(10, 10))
        plt.imshow(seg_map, cmap='jet')
        plt.axis('off')
        plt.savefig(os.path.join(results_dir, f"{base_name}_prediction.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Save overlay
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(seg_map, alpha=0.5, cmap='jet')
        plt.axis('off')
        plt.savefig(os.path.join(results_dir, f"{base_name}_overlay.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Results saved to {results_dir} directory.")
    
    return seg_map, iou

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print("Starting SAM2 segmentation with manual correction capability...")
    try:
        seg_map, iou = run_segmentation_with_manual_correction()
        print("Segmentation completed.")
    except Exception as e:
        import traceback
        print(f"Error during segmentation: {e}")
        traceback.print_exc()