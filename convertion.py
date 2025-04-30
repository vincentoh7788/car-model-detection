import os
import pandas as pd
from PIL import Image
# Load the CSV file
names_file = 'names.csv'
with open(names_file, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

print(class_names)
columns = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']

# Create output directory if it doesn't exist
def convert_annotation(csv_file,image_dir,output_dir):
    df = pd.read_csv(csv_file,header=None, names=columns)
    for _, row in df.iterrows():
        filename = row['filename']
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        label = row['label'] - 1
        class_name = class_names[label]
        image_path = os.path.join(image_dir, class_name, filename)
        if not os.path.exists(image_path):
            print(f"Warning: Image {filename} not found in {image_dir}. Skipping...")
            continue
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Convert to YOLO format
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        # Prepare YOLO annotation line
        yolo_line = f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

        # Write to corresponding TXT file
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, "a") as f:
            f.write(yolo_line)

train_images_dir = './dataset/car_data/car_data/train'
train_output_dir = './dataset/labels/train'
val_images_dir = './dataset/car_data/car_data/test'
val_output_dir = './dataset/labels/val'

import shutil

def gather_images(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg')):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)
                print(f"Copying: {src_path} -> {dst_path}")  # Debug statement
                shutil.copy2(src_path, dst_path)


#gather_images(train_images_dir, './dataset/images/train')
#gather_images(val_images_dir, './dataset/images/val')

#convert_annotation('./dataset/anno_train.csv',train_images_dir, train_output_dir)
#convert_annotation('./dataset/anno_test.csv',val_images_dir, val_output_dir)


