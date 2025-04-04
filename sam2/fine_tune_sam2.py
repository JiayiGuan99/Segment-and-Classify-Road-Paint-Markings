import os
import random
import pandas as pd
import cv2
import torch
import torch.nn.utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import albumentations as A
from sklearn.model_selection import train_test_split
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Path to the dataset folder
data_dir = "D:/VCLab2/final_project/dataset"
images_dir = os.path.join(data_dir, "sat_pave_dataset/selection_org")
masks_dir = os.path.join(data_dir, "sat_pave_dataset/selection_label")

# Load the train.csv file
train_df = pd.read_csv(os.path.join(data_dir, "sat_pave_dataset/train.csv"))

# Split the data into two halves: one for training and one for testing
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Prepare the training data list
train_data = []
for index, row in train_df.iterrows():
   image_name = row['ImageId']
   mask_name = row['MaskId']

   # Append image and corresponding mask paths
   train_data.append({
       "image": os.path.join(images_dir, image_name),
       "annotation": os.path.join(masks_dir, mask_name)
   })

# Prepare the testing data list
test_data = []
for index, row in test_df.iterrows():
   image_name = row['ImageId']
   mask_name = row['MaskId']

   # Append image and corresponding mask paths
   test_data.append({
       "image": os.path.join(images_dir, image_name),
       "annotation": os.path.join(masks_dir, mask_name)
   })

# Define data augmentation pipeline
def get_augmentation():
    return A.Compose([
        # Spatial transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        
        # Color transforms - Enhanced to preserve red features
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=0.8),  # Less aggressive hue shift to preserve reds
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),  # Better preserve color balance
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        ], p=0.3),
        
        # Weather/lighting simulation
        A.OneOf([
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=4, p=0.3),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, p=0.3),
        ], p=0.2),
        
        # Specific to satellite imagery
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),  # Contrast Limited Adaptive Histogram Equalization
    ], p=0.8)

# Data augmentation pipeline
augmentation = get_augmentation()

# Function to detect red regions in an image (potential pedestrian crossings)
def detect_red_regions(img):
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Define range for red color
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine masks
    red_mask = cv2.bitwise_or(mask1, mask2)
    return red_mask

def read_batch(data, visualize_data=False, apply_augmentation=True):
    # Select a random entry
    ent = data[np.random.randint(len(data))]

    # Get full paths
    img = cv2.imread(ent["image"])[..., ::-1]  # Convert BGR to RGB
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # Read annotation as grayscale

    if img is None or ann_map is None:
        print(f"Error: Could not read image or mask from path {ent['image']} or {ent['annotation']}")
        return None, None, None, 0, None

    # Resize image and mask
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])  # Scaling factor
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # Detect red regions (potential pedestrian crossings) before augmentation
    original_red_mask = detect_red_regions(img)

    # Apply data augmentation
    if apply_augmentation:
        augmented = augmentation(image=img, mask=ann_map)
        img = augmented['image']
        ann_map = augmented['mask']

    # Initialize a single binary mask
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
    points = []

    # Get binary masks and combine them into a single mask
    inds = np.unique(ann_map)[1:]  # Skip the background (index 0)
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)  # Create binary mask for each unique index
        binary_mask = np.maximum(binary_mask, mask)  # Combine with the existing binary mask

    # Detect red regions in the augmented image
    red_mask = detect_red_regions(img)
    
    # Create a weight map - give higher weights to red regions
    weight_map = np.ones_like(binary_mask, dtype=np.float32)
    
    # Increase weights for areas that are both in the binary mask and red regions
    red_and_mask = np.logical_and(binary_mask > 0, red_mask > 0).astype(np.float32)
    # Apply a substantial weight increase for pedestrian crossings (3x weight)
    weight_map = weight_map + red_and_mask * 2.0  
    
    # Erode the combined binary mask to avoid boundary points
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)

    # Get all coordinates inside the eroded mask and choose a random point
    coords = np.argwhere(eroded_mask > 0)
    if len(coords) > 0:
        for _ in inds:  # Select as many points as there are unique labels
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([yx[1], yx[0]])

    points = np.array(points)

    if visualize_data:
        # Plotting the images, points, and weights
        plt.figure(figsize=(15, 10))

        # Original Image
        plt.subplot(2, 3, 1)
        plt.title('Original Image')
        plt.imshow(img)
        plt.axis('off')

        # Segmentation Mask (binary_mask)
        plt.subplot(2, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')

        # Mask with Points in Different Colors
        plt.subplot(2, 3, 3)
        plt.title('Binarized Mask with Points')
        plt.imshow(binary_mask, cmap='gray')

        # Plot points in different colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i+1}')

        plt.axis('off')
        
        # Red regions detection
        plt.subplot(2, 3, 4)
        plt.title('Detected Red Regions')
        plt.imshow(red_mask, cmap='gray')
        plt.axis('off')
        
        # Weight map visualization
        plt.subplot(2, 3, 5)
        plt.title('Weight Map (Brighter = Higher Weight)')
        plt.imshow(weight_map, cmap='hot')
        plt.colorbar()
        plt.axis('off')
        
        # Overlay of red regions on the original image
        plt.subplot(2, 3, 6)
        overlay = img.copy()
        overlay[red_mask > 0] = [255, 0, 0]  # Highlight detected red areas
        plt.title('Red Regions Overlay')
        plt.imshow(overlay)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    binary_mask = np.expand_dims(binary_mask, axis=-1)  # Now shape is (H, W, 1)
    binary_mask = binary_mask.transpose((2, 0, 1))
    weight_map = np.expand_dims(weight_map, axis=0)  # Shape to match binary_mask
    points = np.expand_dims(points, axis=1)

    # Return the image, binarized mask, points, number of masks, and weight map
    return img, binary_mask, points, len(inds), weight_map

# Visualize the data with augmentation
Img1, masks1, points1, num_masks, weight_map1 = read_batch(train_data, visualize_data=True, apply_augmentation=True)

# Model paths
sam2_checkpoint = "D:/VCLab2/final_project/dataset/sat_pave_dataset/sam2.1_hiera_small.pt"
model_cfg = "D:/VCLab2/final_project/dataset/sat_pave_dataset/sam2.1_hiera_s.yaml"

# Initialize the model
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Train mask decoder
predictor.model.sam_mask_decoder.train(True)

# Train prompt encoder
predictor.model.sam_prompt_encoder.train(True)

# Configure optimizer with weight decay
optimizer = torch.optim.AdamW(
    params=predictor.model.parameters(),
    lr=0.0001,
    weight_decay=1e-4
)

# Mixed precision for faster training
scaler = torch.cuda.amp.GradScaler()

# Training settings
NO_OF_STEPS = 40000
FINE_TUNED_MODEL_NAME = "fine_tuned_sam2_weighted_loss"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NO_OF_STEPS, eta_min=1e-6)
accumulation_steps = 4  # Number of steps to accumulate gradients before updating

# Function to calculate class-weighted loss
def weighted_loss(prd_mask, gt_mask, weight_map):
    # Convert tensors to device if needed
    if isinstance(weight_map, np.ndarray):
        weight_map = torch.tensor(weight_map, dtype=torch.float32).cuda()
    
    # Dice loss with weights
    intersection = (gt_mask * prd_mask * weight_map).sum((1, 2))
    weighted_dice_loss = 1 - (2 * intersection) / ((gt_mask * weight_map).sum((1, 2)) + (prd_mask * weight_map).sum((1, 2)) + 1e-8)
    
    # BCE loss with weights
    weighted_bce_loss = (-weight_map * gt_mask * torch.log(prd_mask + 1e-8) - 
                           weight_map * (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-8)).mean((1, 2))
    
    # Combined weighted loss
    return 0.5 * weighted_dice_loss.mean() + 0.5 * weighted_bce_loss.mean()

# Training loop with augmentation and weighted loss
best_iou = 0
for step in range(1, NO_OF_STEPS + 1):
    with torch.cuda.amp.autocast():
        # Get augmented data batch with weight map
        image, mask, input_point, num_masks, weight_map = read_batch(train_data, visualize_data=False, apply_augmentation=True)
        if image is None or mask is None or num_masks == 0:
            continue

        input_label = np.ones((num_masks, 1))
        if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
            continue

        if input_point.size == 0 or input_label.size == 0:
            continue

        # Process image
        predictor.set_image(image)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True
        )
        
        if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
            continue

        # Forward pass
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None,
        )

        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

        # Calculate losses with weight map
        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        
        # Use weighted loss function
        seg_loss = weighted_loss(prd_mask, gt_mask, weight_map)
        
        # IoU calculation and score loss
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        union = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
        iou = inter / (union + 1e-8)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        
        # Total loss - weighted segmentation loss plus score loss
        loss = seg_loss + score_loss * 0.05

        # Apply gradient accumulation
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

    # Gradient accumulation - only update weights after accumulation_steps
    if step % accumulation_steps == 0:
        # Clip gradients to prevent explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
        
        # Update weights
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # Update learning rate
    scheduler.step()

    # Save checkpoints periodically
    if step % 500 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{FINE_TUNED_MODEL_NAME}_{step}.torch")
        torch.save(predictor.model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint at step {step} to {checkpoint_path}")

    # Calculate running average IoU
    if step == 1:
        mean_iou = 0
    mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

    # Save best model based on IoU
    current_iou = np.mean(iou.cpu().detach().numpy())
    if current_iou > best_iou:
        best_iou = current_iou
        best_model_path = os.path.join(CHECKPOINT_DIR, f"{FINE_TUNED_MODEL_NAME}_best.torch")
        torch.save(predictor.model.state_dict(), best_model_path)
        print(f"New best model saved with IoU: {best_iou:.4f}")

    # Log progress
    if step % 50 == 0:
        print(f"Step {step}:\t Loss: {loss.item() * accumulation_steps:.4f}, Accuracy (IoU): {mean_iou:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Save final model
final_model_path = os.path.join(CHECKPOINT_DIR, f"{FINE_TUNED_MODEL_NAME}_final.torch")
torch.save(predictor.model.state_dict(), final_model_path)
print(f"Training completed. Final model saved to {final_model_path}")

# Function for inference
def read_image(image_path, mask_path):
    img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
    mask = cv2.imread(mask_path, 0)
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    return img, mask

def get_points(mask, num_points):
    points = []
    coords = np.argwhere(mask > 0)
    if len(coords) > 0:
        for i in range(min(num_points, len(coords))):
            yx = np.array(coords[np.random.randint(len(coords))])
            points.append([[yx[1], yx[0]]])
    return np.array(points)

def evaluate_model(model_path, test_data, visualize_samples=10):
    # Initialize test metrics
    total_iou = 0
    total_dice = 0
    pedestrian_crossing_iou = 0
    pedestrian_crossing_count = 0
    
    # Create results directory
    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the model
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.model.load_state_dict(torch.load(model_path))
    predictor.model.eval()
    
    # Process all test data
    print(f"Evaluating model on full test dataset ({len(test_data)} samples)...")
    
    # Create a CSV file to store metrics
    csv_path = os.path.join(results_dir, "evaluation_metrics.csv")
    with open(csv_path, 'w') as f:
        f.write("sample_id,image_path,iou,dice,has_pedestrian_crossing,pedestrian_crossing_iou\n")
    
    # Select random samples for visualization
    if visualize_samples > 0:
        viz_indices = random.sample(range(len(test_data)), min(visualize_samples, len(test_data)))
    else:
        viz_indices = []
    
    # Process each sample
    for i, entry in enumerate(test_data):
        image_path = entry['image']
        mask_path = entry['annotation']
        
        # Load image and mask
        image, gt_mask = read_image(image_path, mask_path)
        
        # Detect red regions (potential pedestrian crossings)
        red_regions = detect_red_regions(image)
        
        # Generate input points
        input_points = get_points(gt_mask, 30)
        if input_points.size == 0:
            print(f"Skipping sample {i+1}/{len(test_data)} - no valid points found")
            continue
            
        # Perform inference
        with torch.no_grad():
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=np.ones([input_points.shape[0], 1])
            )
        
        # Process predictions
        np_masks = masks[:, 0]
        np_scores = scores[:, 0]
        sorted_indices = np.argsort(np_scores)[::-1]
        sorted_masks = np_masks[sorted_indices]
        
        # Combine masks to create segmentation map
        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
        occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
        
        for j in range(sorted_masks.shape[0]):
            mask = sorted_masks[j]
            if (mask * occupancy_mask).sum() / (mask.sum() + 1e-8) > 0.15:
                continue
                
            mask_bool = mask.astype(bool)
            mask_bool[occupancy_mask] = False
            seg_map[mask_bool] = j + 1
            occupancy_mask[mask_bool] = True
        
        # Calculate metrics
        binary_gt = (gt_mask > 0).astype(np.uint8)
        binary_pred = (seg_map > 0).astype(np.uint8)
        
        intersection = np.logical_and(binary_gt, binary_pred).sum()
        union = np.logical_or(binary_gt, binary_pred).sum()
        iou = intersection / (union + 1e-8)
        
        dice = (2 * intersection) / (binary_gt.sum() + binary_pred.sum() + 1e-8)
        
        total_iou += iou
        total_dice += dice
        
        # Flag for pedestrian crossing presence
        has_pedestrian_crossing = False
        ped_crossing_iou = 0.0
        
        # Calculate metrics for pedestrian crossings specifically
        if np.sum(red_regions) > 0:
            # Create a ground truth mask for red regions
            red_gt = np.logical_and(binary_gt, red_regions > 0).astype(np.uint8)
            # Get prediction for red regions
            red_pred = np.logical_and(binary_pred, red_regions > 0).astype(np.uint8)
            
            if np.sum(red_gt) > 0:  # Only calculate if there are actual red crossings in ground truth
                has_pedestrian_crossing = True
                red_intersection = np.logical_and(red_gt, red_pred).sum()
                red_union = np.logical_or(red_gt, red_pred).sum()
                red_iou = red_intersection / (red_union + 1e-8)
                
                pedestrian_crossing_iou += red_iou
                pedestrian_crossing_count += 1
                ped_crossing_iou = red_iou
                
                print(f"Sample {i+1}/{len(test_data)} - IoU: {iou:.4f}, Dice: {dice:.4f}, Ped. Crossing IoU: {red_iou:.4f}")
        else:
            print(f"Sample {i+1}/{len(test_data)} - IoU: {iou:.4f}, Dice: {dice:.4f}")
            
        # Write metrics to CSV
        with open(csv_path, 'a') as f:
            f.write(f"{i},{image_path},{iou:.6f},{dice:.6f},{int(has_pedestrian_crossing)},{ped_crossing_iou:.6f}\n")
        
        # Visualization for selected samples
        if i in viz_indices:
            plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 3, 1)
            plt.title(f'Test Image {i+1}')
            plt.imshow(image)
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title(f'Original Mask (IoU: {iou:.4f})')
            plt.imshow(gt_mask, cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title(f'Segmentation (Dice: {dice:.4f})')
            plt.imshow(seg_map, cmap='jet')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"test_result_{i+1}.png"))
            plt.close()
    
    # Calculate and save average metrics
    avg_iou = total_iou / len(test_data)
    avg_dice = total_dice / len(test_data)
    
    results_summary = f"""
    Evaluation Summary:
    -------------------
    Total samples: {len(test_data)}
    Average IoU: {avg_iou:.4f}
    Average Dice: {avg_dice:.4f}
    """
    
    # Calculate pedestrian crossing specific metrics
    if pedestrian_crossing_count > 0:
        avg_pedestrian_crossing_iou = pedestrian_crossing_iou / pedestrian_crossing_count
        results_summary += f"""
    Pedestrian Crossing Metrics:
    ---------------------------
    Total crossings found: {pedestrian_crossing_count}
    Average Pedestrian Crossing IoU: {avg_pedestrian_crossing_iou:.4f}
    """
    else:
        results_summary += "\nNo pedestrian crossings found in test samples."
    
    # Save summary to file
    with open(os.path.join(results_dir, "evaluation_summary.txt"), 'w') as f:
        f.write(results_summary)
    
    print(results_summary)
    return avg_iou, avg_dice

# Evaluate the model on the full test dataset
print(f"Total test samples: {len(test_data)}")
best_model_path = os.path.join(CHECKPOINT_DIR, f"{FINE_TUNED_MODEL_NAME}_best.torch")
evaluate_model(best_model_path, test_data, visualize_samples=20)