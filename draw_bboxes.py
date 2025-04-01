import os
import cv2
from generate_new_bboxes import draw_yolo_bboxes, load_yolo_bboxes

def process_images(input_dir, output_dir):
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all images in the images directory
    for image_name in os.listdir(images_dir):
        if image_name.endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(images_dir, image_name)
            label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")
            output_path = os.path.join(output_dir, image_name)

            # Check if the corresponding label file exists
            if not os.path.exists(label_path):
                print(f"Label file not found for {image_name}, skipping.")
                continue

            # Load image and bounding boxes
            image = cv2.imread(image_path)
            bbox = load_yolo_bboxes(label_path)

            # Draw bounding boxes on the image
            result_img = draw_yolo_bboxes(image, bbox)

            # Save the annotated image
            cv2.imwrite(output_path, result_img)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_directory = "./fisheye2_person_dataset/train"          # Directory containing images/ and labels/
    output_directory = "./fisheye2_person_dataset/annotated"     # Directory to save annotated images

    process_images(input_directory, output_directory)
    print("Batch processing completed.")
