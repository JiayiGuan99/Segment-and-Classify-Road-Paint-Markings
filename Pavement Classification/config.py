# config for models

import torch

class segformerConfig:
    def __init__(self):
        # data path 
        self.image_dir = "out_put/images"
        self.mask_dir = "out_put/masks"
        self.split_file = "splits/train.txt"
        self.val_split_file = "splits/val.txt"
        self.test_split_file = "splits/test.txt"
        self.label_map_path = "out_put/label_mapping.json"

        # training parameters
        self.batch_size = 4
        self.num_epoch = 20
        self.learning_rate = 5e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model name
        self.model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
        # b0: nvidia/segformer-b0-finetuned-ade-512-512
        # b2: nvidia/segformer-b2-finetuned-ade-512-512

"""
class SwinUnetConfig:
    def __init__(self):
        # data path
        self.image_dir = "out_put/images"
        self.mask_dir = "out_put/masks"
        self.split_file = "splits/train.txt"
        self.val_split_file = "splits/val.txt"
        self.label_map_path = "out_put/label_mapping.json"

        # training parameters
        self.batch_size = 4
        self.num_epochs = 25
        self.learning_rate = 3e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model name
        self.model_name = "swin-unet-finetune" # 这个记得改一下"
"""
