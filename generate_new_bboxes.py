import cv2
import numpy as np
from fisheye_transformation import apply_fisheye

def yolo_to_absolute(bboxes, img_w, img_h):
    abs_boxes = []
    for cls_id, x, y, bw, bh in bboxes:
        x_center = x * img_w
        y_center = y * img_h
        box_w = bw * img_w
        box_h = bh * img_h

        x1 = x_center - box_w / 2
        y1 = y_center - box_h / 2
        x2 = x_center + box_w / 2
        y2 = y_center + box_h / 2

        abs_boxes.append((cls_id, x1, y1, x2, y2))
    return abs_boxes

def generate_bbox_mask(image_shape, bbox_absolute):
 
    masks = []
    for _, x1, y1, x2, y2 in bbox_absolute:
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=-1)
        masks.append(mask)
    return masks

def find_fisheye_yolo_bbox(fisheye_mask, img_w, img_h, circle_center, circle_radius):

    # Find countours
    contours, _ = cv2.findContours(fisheye_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 2) Find countour with the greater area
    biggest_contour = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(biggest_contour)

    # 3) Clip a la zona visible (opcional)
    x_min = np.clip(x, circle_center[0] - circle_radius, circle_center[0] + circle_radius)
    y_min = np.clip(y, circle_center[1] - circle_radius, circle_center[1] + circle_radius)
    x_max = np.clip(x + w_box, circle_center[0] - circle_radius, circle_center[0] + circle_radius)
    y_max = np.clip(y + h_box, circle_center[1] - circle_radius, circle_center[1] + circle_radius)

    if (x_max - x_min) < 5 or (y_max - y_min) < 5:
        return None

    # 5) Normalized YOLO format
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    w_bbox = (x_max - x_min) / img_w
    h_bbox = (y_max - y_min) / img_h

    return [x_center, y_center, w_bbox, h_bbox]

def draw_yolo_bboxes(image, bboxes, color=(0, 255, 0), thickness=2):

    h, w = image.shape[:2]
    img_copy = image.copy()

    for cls_id, x, y, bw, bh in bboxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img_copy, f"Class {cls_id}", (x1, max(y1 - 5, 0)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_copy

def load_yolo_bboxes(txt_path):

    bboxes = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            values = list(map(float, line.strip().split()))
            if len(values) == 5:  # Check correct format
                bboxes.append(values)
    return bboxes

def generate_new_bboxes(image_shape, bbox_path, map_x, map_y, output_txt_path):

    h, w = image_shape[:2]

    bboxes_yolo = load_yolo_bboxes(bbox_path)
    if not bboxes_yolo:
        print("❌ Error: No bounding boxes found in the file.")
        return None

    circle_center = (w // 2, h // 2)
    circle_radius = min(circle_center)

    bboxes_absolute = yolo_to_absolute(bboxes_yolo, w, h)

    bbox_masks = generate_bbox_mask(image_shape, bboxes_absolute)
    transformed_bboxes = []

    for i, bbox_abs in enumerate(bboxes_absolute):
        cls_id = int(bbox_abs[0])
        distorted_mask = apply_fisheye(
            bbox_masks[i],
            map_x, map_y,
            interpolation=cv2.INTER_NEAREST
        )

        new_bbox = find_fisheye_yolo_bbox(distorted_mask, w, h, circle_center, circle_radius)
        if new_bbox:
            transformed_bboxes.append([cls_id] + new_bbox)
        else:
            print(f"⚠ Warning: Bbox de {bboxes_yolo[i]} se descartó al no estar en área visible.")

    with open(output_txt_path, "w") as f:
        for bbox in transformed_bboxes:
            f.write(" ".join(f"{val:.6f}" for val in bbox) + "\n")

    print(f"✅ Bounding boxes saved to: {output_txt_path}")
    return transformed_bboxes
