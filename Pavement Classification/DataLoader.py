"""
step1. load tiff and shp data 
step2. transform the crs: tifff - 4326; shp - 26910
step3. cut shp areas in tiff and plt
step4. resize the input in to 224*224 patch

"""

import rasterio
import geopandas as gpd
from rasterio.crs import CRS
from rasterio.plot import show
import matplotlib.pyplot as plt
import rasterio.windows
from shapely.geometry import box
import os
import json
import cv2
from rasterio.features import rasterize
import numpy as np
import torch 
from torch.utils.data import Dataset
import albumentations as A 
from albumentations.pytorch import ToTensorV2

class DataLoader():
    def __init__(self, tiff_path = "./dataset/NWM_INT_PAINT.tiff", shp_path = "./dataset/NWM_paint/NWM_paint/paint.shp"):
 
        self.tiff_path = tiff_path
        self.shp_path = shp_path

        self.read_data() # 为了后面能直接用self.shp_data变量，而不用外部调用

    def read_data(self):
        self.tiff_data, self.tiff_profile, self.tiff_crs = self._load_tiff()
        self.shp_data, self.shp_crs = self._load_shp()
        
        if self.tiff_crs != self.shp_crs:
            self.shp_data = self._convert_shp_crs()
    
    def _load_tiff(self):
        with rasterio.open(self.tiff_path) as src: 
            tiff_data = src.read(1) # 读第一波段的数据, 提供的数据只有一个波段，就是高程数据
            tiff_profile = src.profile
            tiff_crs = src.crs
        return tiff_data, tiff_profile, tiff_crs

    def _load_shp(self):
        shp_data = gpd.read_file(self.shp_path) # GeoDataFrame,有geopandas的属性
        shp_crs = shp_data.crs
        return shp_data, shp_crs

    def _convert_shp_crs(self):
        converted_shp = self.shp_data.to_crs(self.tiff_crs)
        return converted_shp
    
    def plot_data(self, window_size = 5000):
        with rasterio.open(self.tiff_path) as src:
            width, height = src.width, src.height
        
        # 从中心点剪裁一个区域画图
            center_x, center_y = width//2, height//2
            x_min, x_max = center_x - window_size//2, center_x + window_size//2
            y_min, y_max = center_y - window_size//2, center_y + window_size//2

            window = rasterio.windows.Window(x_min, y_min, window_size, window_size)
            small_tiff = src.read(1, window = window)
            transform = src.window_transform(window)

        bbox = box(transform.c, transform.f, transform.c + window_size*transform.a, transform.f + window_size*transform.e)
        clipped_shp = self.shp_data[self.shp_data.intersects(bbox)]
        fig, ax = plt.subplots(figsize = (8,6))
        ax.imshow(small_tiff, cmap = "terrain", extent = (
            transform.c,  # 左上角 x 坐标（实际地理坐标）
            transform.c + window_size * transform.a,  # 右下角 x 坐标
            transform.f + window_size * transform.e,  # 右下角 y 坐标（注意 transform.e 通常是负的）
            transform.f  # 左上角 y 坐标
        ))
        clipped_shp.plot(ax=ax, color="red", linewidth=0.5)
        plt.title("region Tiff & Shp check")
        plt.show()

class SegmentationPreprocessor:
    def __init__(self, data_loader, patch_size = 512, save_dir = "out_put"):
        self.tiff_data = data_loader.tiff_data ## tiff_data本身绑定在DataLoader这个类的self上
        self.tiff_profile = data_loader.tiff_profile
        self.transform = self.tiff_profile["transform"]   
        self.crs = self.tiff_profile["crs"]
        self.shp_data = data_loader.shp_data
        
        self.patch_size = patch_size
        self.save_dir = save_dir
        self.image_dir = os.path.join(save_dir, "images")
        self.mask_dir = os.path.join(save_dir, "masks")
        self.label_map_path = os.path.join(save_dir, "label_mapping.json")
        
        os.makedirs(self.image_dir, exist_ok = True)
        os.makedirs(self.mask_dir, exist_ok= True)

        self.label2id = self._build_label_map()

    def _build_label_map(self, freq=5):
        ## 首先对type数据进行预处理，处理逻辑写在README了

        label_clean_map = {
            "arow": "arrow", "ar":"arrow",
            "biike": "bike", "bikew": "bike", "bikwe": "bike", "bikw": "bike", "bike lane": "bike",
            "bus only": "bus",
            "bmp": "bump", "bumpp": "bump",
            "cw": "crosswalk", "cw'": "crosswalk", "cross": "crosswalk",
            "ds": "double_solid",
            "dsb": "double_solid_broken",
            "hashy": "hash", "hahs": "hash",
            "pedesterian": "pedestrian",
            "do not stop": "do_not_stop", "do nots stop": "do_not_stop",
            "sb": "single_broken",
            "ss": "single_solid", "ssl": "single_solid", "ssy": "single_solid", "solid": "ssingle_solid",
            "sl": "stopline", "stop line": "stopline", 
            "p": "parking"
        }
        # missing: rod, hov

        raw_labels = self.shp_data["type"].dropna().astype(str).str.lower()
        clean_labels = raw_labels.apply(lambda x: label_clean_map.get(x,x)) #.get(x,x)在字典中寻找x对应的value，如果有的话就返回对应value, 如果没有的话就返回x自身
        
        # 统计频数，只返回大于设定频数的值
        label_count = clean_labels.value_counts()
        freq_labels = label_count[label_count>=freq].index.tolist()

        clean_labels = clean_labels.apply(lambda x: x if x in freq_labels else None)
        self.shp_data["type"] = clean_labels
        
        unique_labels = sorted(set(label for label in clean_labels if label is not None))
        label2id = {label: idx+1 for idx, label in enumerate(unique_labels)} ## unique_label有none要预处理, 为了满足corssentropy, label从0开始

        with open(self.label_map_path, "w") as f:
            json.dump(label2id, f) # dump：把python对象(e.g. 字典)以json的格式写入文件中
        return label2id

    def _rasterize_mask(self, shapes, out_shape, transform):
        # print("Rasterizing shape count:", len(shapes))
        # if shapes:
        #     print("First shape geometry:", shapes[0][0])
        #     print("First shape label ID:", shapes[0][1])

        mask = rasterize(
            shapes, 
            out_shape=out_shape,
            transform=transform,
            fill=0, # 除了shape外的background都是0
            dtype="uint8",
            all_touched=True
        )

        # print("Mask unique values:", np.unique(mask))
        return mask
    
    def generate_patches(self):
        height, width = self.tiff_data.shape
        count = 0 # 用于生成patch编号

        for y in range(0, height, self.patch_size):
            for x in range(0, width, self.patch_size):

                window = rasterio.windows.Window(x,y,self.patch_size,self.patch_size) # 创建了一个patch size的矩形窗口
                patch = self.tiff_data[y:y+self.patch_size, x:x+self.patch_size] # 从tiff中汲截取一个patch, 和window的范围对应，这个数据会输入模型进行训练

                # 如果的image的size 不是512*512的话，跳过，因为在test中检查了，只有 三张是不符合的
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    # print(f"Skipping patch {count:04d} due to image size: {patch.shape}") # debug
                    continue

                # calculate geospatial bounds of this patch
                patch_transform = rasterio.windows.transform(window, self.transform) # 这里用到了上面定义的矩形窗口，并对这个区域的范围进行了affine变换(像素->地理坐标)
                patch_bounds = rasterio.windows.bounds(window, self.transform) # 返回的是(x_min, y_min, x_max, y_max)
                patch_box = box(*patch_bounds) #用来构造一个shapely的矩形polygon区域，用来和shp的集合做交集判断，判断就在下面

                # Clip SHP to patch bounds
                clipped = self.shp_data[self.shp_data.intersects(patch_box)].copy() 
                if clipped.empty: # 如果这个patch_box中不包含shp文件中标记的road_marking的话，就略过它
                    continue 
                # convert geometries to (geometry, class_id) tuples
                ## 这里是在对上面得到的clipped处理，clipped是GeoDataFrame
                shapes = [
                    (geom, self.label2id[row["type"]])
                    for _, row in clipped.iterrows() # .iterow(): pandas & gpd中常见的按行迭代的方式，得到(index, row)
                    if row["type"] is not None and row["type"] in self.label2id
                    for geom in row.geometry.geoms # 取出每个Multi Polygon里的所有polygon
                    if hasattr(row.geometry, "geoms") # 检查row是否有geom这个属性; 列表推导式中，if在for后面是推导式
                ] if clipped.geometry.iloc[0].geom_type == "MultiPolygon" else [
                    (row.geometry, self.label2id[row["type"]]) 
                    for _, row in clipped.iterrows()
                    if row["type"] is not None and row["type"] in self.label2id # row是geopandas中的一个series, 属性(type, geometry)
                ] # 先判断是CLIPPED是不是Multipolygen类型（即一个对象里包含多个polygon，比如几个箭头的组合），如果是的话，就展开每个小polygon

                mask = self._rasterize_mask(shapes, out_shape=(self.patch_size, self.patch_size), transform=patch_transform) # 输入矢量图，输出二维像素Mask图

                # Skip patches without any label
                if mask.max() == 0:
                    continue 

                # save images and masks
                img_name = f"patch_{count:04d}.png"
                mask_name = f"patch_{count:04d}_mask.png"

                cv2.imwrite(os.path.join(self.image_dir, img_name), patch)
                cv2.imwrite(os.path.join(self.mask_dir, mask_name), mask)
                count += 1
        print(f"Generated {count} image-mask pairs in '{self.save_dir}'")