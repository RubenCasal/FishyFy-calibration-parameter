import cv2
import numpy as np

def resize_to_square(img: np.ndarray) -> np.ndarray:

    size = max(img.shape[:2])
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

def create_fisheye_mapping(h, w, K, D):
    K[0,2] = w/2.0
    K[1,2] = h /2.0

    print(K)
    R = np.eye(3, dtype=np.float32)
   
    scale = 2 
    K[0, 0] *= scale  
    K[1, 1] *= scale  

    # Generate grid for the distorted image
    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    pts_distort = np.stack([xv.flatten(), yv.flatten()], axis=-1).astype(np.float32)

    pts_distort_reshaped = pts_distort.reshape(-1, 1, 2)
    
    # Compute the distorted map
    pts_ud = cv2.fisheye.undistortPoints(pts_distort_reshaped, K, D, R=R, P=K)

    # Reshape the map to image size
    map_x = pts_ud[:, 0, 0].reshape(h, w)
    map_y = pts_ud[:, 0, 1].reshape(h, w)

    return map_x, map_y


def apply_fisheye(image, map_x, map_y, interpolation=cv2.INTER_LINEAR):
   
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

def create_LUT_table(image, K ,D):

    h, w = image.shape[:2]
    map_x, map_y = create_fisheye_mapping(h, w, K, D)
    return map_x, map_y
