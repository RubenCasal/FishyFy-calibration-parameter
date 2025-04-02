
from process_yolo_dataset import process_yolo_subset, update_yaml, get_LUT
import os
import time
import numpy as np

if __name__ == "__main__":
    DATASET_DIR = "./person_dataset"
    OUTPUT_DIR = "./fisheye2_person_dataset"
    SUBSETS = ['train',  'test', 'valid']
   
    K = np.array([[284.509100, 0, 2.0],
              [0, 282.941856, 2.0],
              [0, 0, 1.000000]], dtype=np.float32)
    

    D = np.array([-0.614216, 0.060412,-0.054711, 0.011151], dtype=np.float32)

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