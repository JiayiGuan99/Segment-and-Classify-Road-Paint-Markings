import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import *
from dataloader import load_dataset, read_batch, read_image
from utils import build_model, get_points, process_predictions, visualize_predictions

def main():
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)
    
    # Load dataset
    train_data, test_data = load_dataset()
    
    # Check the data by visualizing a sample
    Img1, masks1, points1, num_masks = read_batch(train_data, visualize_data=True)
    
    # Build the SAM2 model
    predictor = build_model()
    
    # Train mask decoder
    predictor.model.sam_mask_decoder.train(True)
    
    # Train prompt encoder
    predictor.model.sam_prompt_encoder.train(True)
    
    # Configure optimizer
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Mix precision - Fixed deprecation warning
    scaler = torch.amp.GradScaler('cuda')
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=STEP_SIZE, 
        gamma=LR_GAMMA
    )
    
    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Training loop
    mean_iou = 0
    best_iou = 0
    
    for step in range(1, NO_OF_STEPS + 1):
        # Fixed deprecation warning
        with torch.amp.autocast('cuda'):
            image, mask, input_point, num_masks = read_batch(train_data, visualize_data=False)
            if image is None or mask is None or num_masks == 0:
                continue
                
            input_label = np.ones((num_masks, 1))
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                continue
                
            if input_point.size == 0 or input_label.size == 0:
                continue
                
            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
            
            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                continue
                
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
            
            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()
            
            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05
            
            # Apply gradient accumulation
            loss = loss / ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
            
            if step % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                predictor.model.zero_grad()
            
            # Save checkpoint every 500 steps
            if step % 500 == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{FINE_TUNED_MODEL_NAME}_{step}.torch")
                torch.save(predictor.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
                
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
            
            if step % 100 == 0:
                print("Step " + str(step) + ":\t", "Accuracy (IoU) = ", mean_iou)
    
    # Save final model
    final_model_path = os.path.join(CHECKPOINT_DIR, f"{FINE_TUNED_MODEL_NAME}_final.torch")
    torch.save(predictor.model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Testing with a random test image
    test_model(predictor, test_data)

def test_model(predictor=None, test_data=None, model_weights=None):
    """Test the trained model on a random image from the test set"""
    # Randomly select a test image from the test_data
    selected_entry = random.choice(test_data)
    image_path = selected_entry['image']
    mask_path = selected_entry['annotation']
    
    # Load the selected image and mask
    image, mask = read_image(image_path, mask_path)
    
    # Generate random points for the input
    num_samples = 30  # Number of points per segment to sample
    input_points = get_points(mask, num_samples)
    
    if predictor is None:
        # Use best model by default
        if model_weights is None:
            model_weights = os.path.join(CHECKPOINT_DIR, f"{FINE_TUNED_MODEL_NAME}_best.torch")
            # If best model doesn't exist, use the default checkpoint
            if not os.path.exists(model_weights):
                model_weights = os.path.join(CHECKPOINT_DIR, f"{FINE_TUNED_MODEL_NAME}_1000.torch")
        
        # Load the fine-tuned model
        predictor = build_model()
        predictor.model.load_state_dict(torch.load(model_weights))
        print(f"Testing with model: {model_weights}")
    
    # Perform inference and predict masks
    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )
    
    # Process the predicted masks
    seg_map = process_predictions(masks, scores)
    
    # Visualize results
    visualize_predictions(image, mask, seg_map)

if __name__ == "__main__":
    main()