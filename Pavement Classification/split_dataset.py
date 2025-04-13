import os 
import random 

def split_dataset(image_dir, out_put_dir = "splits", train_ratio = 0.7, val_ratio = 0.2, test_ratio=0.1, seed = 42):
    """这个函数只需要运行一次来生成划分文件"""
    os.makedirs(out_put_dir, exist_ok = True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    random.seed(seed)
    random.shuffle(image_files)

    total  = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    with open(os.path.join(out_put_dir, "train.txt"), "w") as f :
        f.writelines([f"{name}\n" for name in train_files])
    
    with open(os.path.join(out_put_dir, "val.txt"), "w") as f:
        f.writelines([f"{name}\n" for name in val_files])

    with open(os.path.join(out_put_dir, "test.txt"), "w") as f:
        f.writelines([f"{name}\n" for name in test_files])
    
    print(f"split complete: {len(train_files)} train / {len(val_files)} val / {len(test_files)} test")


if __name__ == "__main__":
    split_dataset("out_put/images")