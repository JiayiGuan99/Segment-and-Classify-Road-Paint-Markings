# Add all helper functions here

import os 
import random 
from collections import Counter
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import wandb

def load_split_file(split_file):
    with open(split_file, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]
    return filenames

def analyze_class_distribution(dataloader):
    counter = Counter()

    for _, mask in tqdm(dataloader, desc="Counting pixels"):
        for m in mask:  # loop over batch dimension
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            m = m.flatten()
            counter.update(m.tolist())

    print("\nClass Distribution:")
    for class_id, count in sorted(counter.items()):
        print(f"Class {class_id}: {count:,} pixels")
    
    wandb.log({"class_pixel_counts": dict(counter)})
    return counter # A dictionary-like object mapping class indices to pixel counts.

def compute_class_weights(counter, num_classes):
    total = sum(counter.values())
    weights = []
    for i in range(num_classes):
        count = counter.get(i, 0)
        if count == 0:
            weights.append(0.0)  # or small value like 1e-6 to avoid zero
        else:
            weights.append(total / count)

    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum()  # normalize to sum to 1 (optional)
    print("\nClass Weights:", weights.numpy())
    wandb.log({"class_weights": {f"class_{i}": w.item() for i, w in enumerate(weights)}})
    return weights #torch.Tensor: A tensor of class weights for use in loss functions.

# generate color and label pairs
def generate_class_colors(num_classes, colormap_name="tab20"): # nipy_spectral
    cmap = plt.get_cmap(colormap_name, num_classes) 

    class_color = {0:(0,0,0)} # background is black, stay unchanged
    for i in range(1, num_classes):
        rgb = tuple(int(c*255) for c in cmap(i)[:3]) # 返回cmap对象的第i个颜色，理论上数据格式是(R,G,B,透明度)
        class_color[i] = rgb

    return class_color

# decode pairs
def decode_segmap(mask, class_colors):
    """
    mask: 模型预测的H*W二维图像
    将label图转化为彩色图像
    """
    h,w = mask.shape
    color_mask = np.zeros((h,w,3), dtype = np.uint8)
    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color
    return color_mask

def plot_class_legend(class_colors, label_map, save = True):
    """
    generate legend for visualizing color and corresponding labels
    """
    id2label = {v:k for k, v in label_map.items()}

    handles = []
    for class_id, color in sorted(class_colors.items()):
        if class_id == 0:
            continue # skipped black background
        label = f"{class_id}: {id2label.get(class_id, 'Unknown')}"
        patch = mpatches.Patch(color = np.array(color)/255.0, label = label)
        handles.append(patch)

    plt.figure(figsize = (12,1))
    plt.axis("off")
    legend = plt.legend(handles = handles, loc = "center", ncol = 4, frameon = False)
    plt.tight_layout()

    if save:
        plt.savefig("label_color_pairs", dpi = 300, bbox_inches = "tight", pad_inches = 0.2)
        print("Legend saved.")
    else:
        plt.show()

class SplitHelper:
    @staticmethod
    # 这个方法用于帮助我们将指定train/val文件中的image和mask pair起来，给RoadMarkingDataset读取
    def get_split_indices(image_dir, split_file):
        """given a split file, return valid image and mask filenames"""
        split_name = load_split_file(split_file)
        image_files = []
        mask_files = []
        for name in split_name:
            image_path = os.path.join(image_dir, name)
            mask_name = name.replace(".png", "_mask.png")
            if os.path.exists(image_path):
                image_files.append(name)
                mask_files.append(mask_name)
            else:
                print(f"skipped missing image: {image_path}")
        return image_files, mask_files

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        # logits: [B,C,H,W]
        # targets: [B,H,W]
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0,3,1,2).float()
        
        intersection = (probs * targets_one_hot).sum(dim = (0,2,3))
        union = probs.sum(dim = (0,2,3)) + targets_one_hot.sum(dim = (0,2,3))

        dice = (2.*intersection+self.smooth)/(union + self.smooth)
        loss = 1 - dice
        return loss.mean()

class ComboLoss(torch.nn.Module):
    def __init__(self, weight_ce=0.2, weight_dice=0.8, class_weights=None, smooth=1e-6):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(smooth=smooth)
        self.w_ce = weight_ce
        self.w_dice = weight_dice

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.w_ce * ce_loss + self.w_dice * dice_loss

