import numpy as np
import torch 
from torch.utils.data import Dataset
import albumentations as A 
from albumentations.pytorch import ToTensorV2
import cv2
import os
from utils import SplitHelper

class RoadMarkingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None, split_file = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        if split_file:
            self.image_files, self.mask_files = SplitHelper.get_split_indices(image_dir, split_file)

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.mask_files = [f.replace(".png", "_mask.png") for f in self.image_files]

        if self.transform is True:
            self.transform = A.Compose([
                A.Resize(512, 512), 
                A.HorizontalFlip(p=0.5), 
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5,0.5,0.5)), 
                ToTensorV2()
            ])
        elif self.transform is False:
            self.transform = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # print(f"[debug] Getting sample idx: {idx}")

        img_path = os.path.join(self.image_dir, self.image_files[idx])
        msk_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # shape: (H,W)
        # print("image after read:", type(image), image.shape) # debug
        image = np.stack([image]*3, axis = -1) # convert to 3 channels: (H,W,3)? 这种处理是否合理？
        # print("image after stack:", type(image), image.shape) # debug
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        # # debug
        # if image is None or mask is None:
        #     print(f"load failed: idx = {idx}, image = {img_path}, mask = {msk_path}")
        #     return None
        
        if image.shape[:2] != mask.shape[:2]:
            print(f"[ERROR] Shape mismatch! image shape: {image.shape}, mask shape: {mask.shape}")
            return None  # 跳过不合法样本

        # 如果传入了的augmentation的transform就用，否则就直接将原始数据转化为的tensor
        if self.transform:
            augmented = self.transform(image=image, mask = mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else: 
            image = A.ToFloat()(image=image)["image"] # to float 32
            # print("image before ToFloat:", type(image), image.shape) # debug
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # [H,W,C] → [C,H,W]
            mask = torch.from_numpy(mask).long() # 把tensor的dtype转为long, which is int64
        return image, mask