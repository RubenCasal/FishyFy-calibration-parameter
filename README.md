# Fisheye Dataset Transformer

![Python](https://img.shields.io/badge/Python-3.10-yellow.svg)  ![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-blue.svg)  ![NumPy](https://img.shields.io/badge/NumPy-1.21.5-orange.svg)

## 📌 Introduction
This repository provides a pipeline to transform a YOLO format dataset, containing images and bounding boxes, into a fisheye-transformed version. It takes into account the camera calibration parameters (intrinsic matrix `K` and distortion coefficients `D`) to generate a fisheye effect and update the bounding boxes accordingly. The final dataset is in the same YOLO format but adapted to the fisheye transformation.

## Original Bounding Box
<p align="center">
<img src="./readme_images/original.jpg" alt="Original Bounding Box" width="400">
</p>

### Fisheye Bounding Box
<p align="center">
    <img src="./readme_images/transformed.jpg" alt="Fisheye Bounding Box" width="400">
</p>

## 📚 Dataset Format
```
Dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/ (optional)
│   ├── images/
│   └── labels/
└── data.yaml
```

## 🚀 How It Works

### 🎥 Fisheye Image Transformation
1. **Square Resize**: The image is resized to a square to maintain isotropic distortion using the `resize_to_square()` function.

2. **LUT (Lookup Table) Computation**:
   - The LUT (Lookup Table) is a precomputed map that translates each pixel's coordinates from the original image to their new positions after applying the fisheye distortion.
   - The fisheye distortion effect is modeled by projecting the image through a non-linear transformation based on the camera's intrinsic matrix `K` and distortion coefficients `D`.
   - The `create_LUT_table()` function calculates two transformation maps, `map_x` and `map_y`, which contain the horizontal and vertical coordinates for each pixel after distortion.
   - These maps allow the fisheye transformation to be efficiently applied to the entire image in a single remapping step, significantly reducing computational cost when processing large datasets.

3. **Applying Fisheye Transformation**:
   - The `apply_fisheye()` function uses the LUT to remap image pixels, creating the fisheye effect.
   - The `crop_black_borders()` function removes any black areas resulting from the transformation.

4. **Bounding Box Transformation**:
   - Bounding boxes in YOLO format (x_center, y_center, width, height) are converted to absolute coordinates using the `yolo_to_absolute()` function.
   - Each bounding box is transformed through a mask-based fisheye effect using `generate_bbox_mask()`.
   - The transformed bounding box is recalculated using `find_fisheye_yolo_bbox()`, which identifies the smallest enclosing rectangle after distortion.
   - Bounding boxes that are too small or completely outside the visible area are discarded to maintain data accuracy.

### 🌀 Parallel Processing
- Utilizes `ProcessPoolExecutor` for multi-core processing, where each image is processed in parallel to speed up the transformation.
- Computes the LUT once and shares it among worker processes.

## 🔧 How to Use

To use this tool, configure the input dataset path, output path, and camera calibration parameters (`K` and `D`). Run the script to process the dataset, and the transformed images with updated bounding boxes will be saved in the output directory.

```python
from process_yolo_dataset import process_yolo_subset, update_yaml, get_LUT
import os
import time
import numpy as np

DATASET_DIR = "./person_dataset"
OUTPUT_DIR = "./fisheye2_person_dataset"
SUBSETS = ['train']

K = np.array([[284.509100, 0, 421.896335],
              [0, 282.941856, 398.100316],
              [0, 0, 1.000000]], dtype=np.float32)

D = np.array([-0.014216, 0.060412, -0.054711, 0.011151], dtype=np.float32)

start_time = time.perf_counter()
map_x, map_y = get_LUT(DATASET_DIR, K, D)
for subset in SUBSETS:
    images_dir = os.path.join(DATASET_DIR, subset, 'images')
    labels_dir = os.path.join(DATASET_DIR, subset, 'labels')
    output_images_dir = os.path.join(OUTPUT_DIR, subset, 'images')
    output_labels_dir = os.path.join(OUTPUT_DIR, subset, 'labels')
    process_yolo_subset(images_dir, labels_dir, output_images_dir, output_labels_dir, map_x, map_y)

input_yaml = os.path.join(DATASET_DIR, 'data.yaml')
output_yaml = os.path.join(OUTPUT_DIR, 'data.yaml')
update_yaml(input_yaml, output_yaml, OUTPUT_DIR)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Total processing time: {elapsed_time:.2f} seconds")
```

## 👀 Visualization Utility

To visualize the transformed images and their bounding boxes, use the following script. It loads an image and its bounding boxes from the output directory, draws the bounding boxes on the image, and saves the result for quick inspection.
```python
from generate_new_bboxes import draw_yolo_bboxes, load_yolo_bboxes
import cv2

image = cv2.imread("fisheye2_person_dataset/train/images/image1.jpg")
labels = load_yolo_bboxes("fisheye2_person_dataset/train/labels/image1.txt")
image_with_boxes = draw_yolo_bboxes(image, labels)
cv2.imwrite("output/image1_preview.jpg", image_with_boxes)
```

## ⚠️ Limitations
- Some information near the image edges may be lost after applying the fisheye transformation.
- The transformation may discard bounding boxes that are too small or outside the fisheye field of view.
- Due to the fisheye effect, some bounding boxes may slightly move out of the visible area.

