import os
import cv2
import yaml
from concurrent.futures import ProcessPoolExecutor
from fisheye_transformation import resize_to_square, create_LUT_table, apply_fisheye
from generate_new_bboxes import (
    yolo_to_absolute,
    generate_bbox_mask,
    find_fisheye_yolo_bbox,
    load_yolo_bboxes,
)
def crop_black_borders(image, bboxes):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("⚠ No meaningful contours found. Returning original image.")
        return image, bboxes

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])

    if cv2.contourArea(contours[0]) < 0.01 * image.shape[0] * image.shape[1]:
        print("⚠ Bounding box so small to consider. Returning original image.")
        return image, bboxes

    cropped_image = image[y:y+h, x:x+w]

    new_bboxes = []
    for bbox in bboxes:
        cls_id, cx, cy, bw, bh = bbox

        abs_cx = cx * image.shape[1]
        abs_cy = cy * image.shape[0]
        abs_bw = bw * image.shape[1]
        abs_bh = bh * image.shape[0]


        new_cx = (abs_cx - x) / w
        new_cy = (abs_cy - y) / h
        new_bw = abs_bw / w
        new_bh = abs_bh / h


        if 0 <= new_cx <= 1 and 0 <= new_cy <= 1:
            new_bboxes.append([cls_id, new_cx, new_cy, new_bw, new_bh])


    if cropped_image.size == 0 or not new_bboxes:
        print("⚠ Image is completely black or without valid bounding boxes.")
        return image, bboxes

    return cropped_image, new_bboxes



def process_single_image(image_file, images_dir, labels_dir, output_images_dir, output_labels_dir, map_x, map_y):
    image_path = os.path.join(images_dir, image_file)
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_file)

    if not os.path.exists(label_path):
        print(f"⚠ El archivo de etiquetas {label_path} no existe. Se omite la imagen {image_file}.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: no se pudo cargar la imagen {image_path}")
        return

    image_square = resize_to_square(image)
    h, w = image_square.shape[:2]

    bboxes_yolo = load_yolo_bboxes(label_path)
    if not bboxes_yolo:
        print(f"❌ No se encontraron etiquetas válidas en el archivo {label_path}")
        return

    bboxes_absolute = yolo_to_absolute(bboxes_yolo, w, h)
    masks = generate_bbox_mask(image_square.shape, bboxes_absolute)

    circle_center = (w // 2, h // 2)
    circle_radius = min(circle_center)

    fisheye_image = apply_fisheye(image_square, map_x, map_y)
    
    new_bboxes = []
    for i, mask in enumerate(masks):
        cls_id = int(bboxes_absolute[i][0])
        distorted_mask = apply_fisheye(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST)
        bbox = find_fisheye_yolo_bbox(distorted_mask, w, h, circle_center, circle_radius)
        if bbox:
            new_bboxes.append([cls_id] + bbox)
        else:
            print(f"⚠ BBox {i} descartada tras la transformación.")

    cropped_image, cropped_bboxes = crop_black_borders(fisheye_image, new_bboxes)
    output_image_path = os.path.join(output_images_dir, image_file)
    cv2.imwrite(output_image_path, cropped_image)

    output_label_path = os.path.join(output_labels_dir, label_file)
    with open(output_label_path, "w") as f:
        for bbox in cropped_bboxes:
            f.write(" ".join(f"{val:.6f}" for val in bbox) + "\n")

    print(f"✅ Procesado {image_file}: imagen y etiquetas guardadas.")

def get_LUT(dataset_dir,K, D):
    images_dir = os.path.join(dataset_dir, 'train', 'images')
    first_image_file = next(
    (entry.name for entry in os.scandir(images_dir) if entry.is_file() and entry.name.endswith(('.jpg', '.png'))),
    None)
    first_image_path = os.path.join(images_dir, first_image_file)
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"❌ Error: no se pudo cargar la imagen {first_image_path}")
        return

    first_image_square = resize_to_square(first_image)
    map_x, map_y = create_LUT_table(first_image_square, K, D)
    return map_x, map_y
    


def process_yolo_subset(images_dir, labels_dir, output_images_dir, output_labels_dir, map_x, map_y):
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = [entry.name for entry in os.scandir(images_dir) if entry.is_file() and entry.name.endswith(('.jpg', '.png'))]

    if not image_files:
        print(f"❌ No se encontraron imágenes en {images_dir}.")
        return

    num_cores = os.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [
            executor.submit(
                process_single_image,
                image_file,
                images_dir,
                labels_dir,
                output_images_dir,
                output_labels_dir,
                map_x,
                map_y
            )
            for image_file in image_files
        ]
        for future in futures:
            future.result()

def update_yaml(input_yaml, output_yaml, output_dir):
    with open(input_yaml, "r") as f:
        data = yaml.safe_load(f)

    data['train'] = os.path.join(output_dir, "train")
    data['val'] = os.path.join(output_dir, "val")
    if 'test' in data:
        data['test'] = os.path.join(output_dir, "test")

    with open(output_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print("✅ Archivo data.yaml actualizado.")
