import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from config import *

def load_dataset():
    """Load and split dataset into training and testing sets"""
    train_df = pd.read_csv(TRAIN_CSV)
    
    # Split the data into training and testing
    train_df, test_df = train_test_split(train_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Prepare the training data list
    train_data = []
    for index, row in train_df.iterrows():
        image_name = row['ImageId']
        mask_name = row['MaskId']
        
        # Append image and corresponding mask paths
        train_data.append({
            "image": os.path.join(IMAGES_DIR, image_name),
            "annotation": os.path.join(MASKS_DIR, mask_name)
        })
    
    # Prepare the testing data list
    test_data = []
    for index, row in test_df.iterrows():
        image_name = row['ImageId']
        mask_name = row['MaskId']
        
        # Append image and corresponding mask paths
        test_data.append({
            "image": os.path.join(IMAGES_DIR, image_name),
            "annotation": os.path.join(MASKS_DIR, mask_name)
        })
    
    return train_data, test_data

def read_batch(data, visualize_data=False):
    """Read and process a random batch from the dataset"""
    # Select a random entry
    ent = data[np.random.randint(len(data))]
    
    # Get full paths
    Img = cv2.imread(ent["image"])[..., ::-1]  # Convert BGR to RGB
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # Read annotation as grayscale
    
    if Img is None or ann_map is None:
        print(f"Error: Could not read image or mask from path {ent['image']} or {ent['annotation']}")
        return None, None, None, 0
    
    # Resize image and mask
    r = np.min([MAX_SIZE / Img.shape[1], MAX_SIZE / Img.shape[0]])  # Scaling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    
    # Initialize a single binary mask
    binary_mask = np.zeros_like(ann_map, dtype=np.uint8)
    points = []
    
    # Get binary masks and combine them into a single mask
    inds = np.unique(ann_map)[1:]  # Skip the background (index 0)
    for ind in inds:
        mask = (ann_map == ind).astype(np.uint8)  # Create binary mask for each unique index
        binary_mask = np.maximum(binary_mask, mask)  # Combine with the existing binary mask
    
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
        # Plotting the images and points
        plt.figure(figsize=(15, 5))
        
        # Original Image
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(Img)
        plt.axis('off')
        
        # Segmentation Mask (binary_mask)
        plt.subplot(1, 3, 2)
        plt.title('Binarized Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')
        
        # Mask with Points in Different Colors
        plt.subplot(1, 3, 3)
        plt.title('Binarized Mask with Points')
        plt.imshow(binary_mask, cmap='gray')
        
        # Plot points in different colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i+1}')
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    binary_mask = np.expand_dims(binary_mask, axis=-1)  # Now shape is (H, W, 1)
    binary_mask = binary_mask.transpose((2, 0, 1))
    points = np.expand_dims(points, axis=1)
    
    # Return the image, binarized mask, points, and number of masks
    return Img, binary_mask, points, len(inds)

def read_image(image_path, mask_path):
    """Read and resize image and mask"""
    img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
    mask = cv2.imread(mask_path, 0)
    
    r = np.min([MAX_SIZE / img.shape[1], MAX_SIZE / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    
    return img, mask