import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_red_regions(img):
    """
    Detect red regions in an image using HSV color space
    Returns a binary mask where red regions are marked as 1
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for red color (two ranges since red wraps around in HSV)
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

def generate_weight_map(img, red_mask, binary_mask=None):
    """
    Generate a weight map giving higher weight to red regions
    If binary_mask is provided, highlights intersection of red regions and mask
    """
    # Create base weight map with all weights set to 1
    weight_map = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    
    # If binary mask is provided, use it for combined weighting
    if binary_mask is not None:
        # Increase weights for areas that are both in the binary mask and red regions
        red_and_mask = np.logical_and(binary_mask > 0, red_mask > 0).astype(np.float32)
        # Apply a substantial weight increase for pedestrian crossings (3x weight)
        weight_map = weight_map + red_and_mask * 2.0
    else:
        # Without mask, just add weight to red regions
        weight_map = weight_map + (red_mask > 0).astype(np.float32) * 2.0
    
    return weight_map

def process_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    print(f"Image loaded successfully. Dimensions: {width}x{height}")
    
    # Create a simple binary mask (for demonstration - could be replaced with actual segmentation)
    # This is just a placeholder - in real usage you'd use your segmentation mask
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    # For demonstration, let's create a simple rectangle mask in the center
    center_h, center_w = height // 2, width // 2
    binary_mask[center_h-50:center_h+50, center_w-50:center_w+50] = 1
    
    # Detect red regions
    red_mask = detect_red_regions(img)
    print(f"Detected {np.sum(red_mask > 0)} pixels in red regions")
    
    # Generate weight map
    weight_map_with_mask = generate_weight_map(img, red_mask, binary_mask)
    weight_map_no_mask = generate_weight_map(img, red_mask)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Original Image
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    # Red mask
    plt.subplot(2, 3, 2)
    plt.title('Detected Red Regions')
    plt.imshow(red_mask, cmap='gray')
    plt.axis('off')
    
    # Binary mask (placeholder)
    plt.subplot(2, 3, 3)
    plt.title('Binary Mask (Example)')
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    
    # Weight map without mask
    plt.subplot(2, 3, 4)
    plt.title('Weight Map (Red Regions Only)')
    plt.imshow(weight_map_no_mask, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    
    # Weight map with mask
    plt.subplot(2, 3, 5)
    plt.title('Weight Map with Mask')
    plt.imshow(weight_map_with_mask, cmap='hot')
    plt.colorbar()
    plt.axis('off')
    
    # Overlay red on original
    plt.subplot(2, 3, 6)
    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    red_overlay = np.zeros_like(overlay)
    red_overlay[red_mask > 0] = [255, 0, 0]
    overlay = cv2.addWeighted(overlay, 0.7, red_overlay, 0.3, 0)
    plt.title('Red Regions Overlay')
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("red_region_detection_result.png")
    plt.show()
    
    return img, red_mask, weight_map_with_mask

if __name__ == "__main__":
    # Process the specific image
    image_path = "D:/VCLab2/final_project/dataset/sat_pave_dataset/selection_org/test_002.png"
    img, red_mask, weight_map = process_image(image_path)
    
    # Save outputs
    cv2.imwrite("detected_red_mask.png", red_mask)
    # Scale weight map to 0-255 for better visualization when saved
    weight_map_normalized = ((weight_map - weight_map.min()) / 
                            (weight_map.max() - weight_map.min()) * 255).astype(np.uint8)
    cv2.imwrite("weight_map.png", cv2.applyColorMap(weight_map_normalized, cv2.COLORMAP_JET))
    
    print("Processing complete. Results saved to disk.")