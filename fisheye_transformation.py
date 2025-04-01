import cv2
import numpy as np

def resize_to_square(img: np.ndarray) -> np.ndarray:

    size = max(img.shape[:2])
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

def create_fisheye_mapping(h, w, strength=0.6):
    
    K = np.array([[284.509100, 0, w/2.0],
              [0, 282.941856, h/2.0],
              [0, 0, 1.000000]], dtype=np.float32)
    

    D = np.array([-0.614216, 0.060412,-0.054711, 0.011151], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    # Scale focal lengths to reduce black borders
    scale = 2  # Reduced scale factor to minimize black areas
    K[0, 0] *= scale  # fx
    K[1, 1] *= scale  # fy

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

def create_LUT_table(image, distortion_strength=0.6):

    h, w = image.shape[:2]
    map_x, map_y = create_fisheye_mapping(h, w,distortion_strength)
    return map_x, map_y
