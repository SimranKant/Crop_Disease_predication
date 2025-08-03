import os
import random
import shutil

# ✅ Set your actual dataset root directory here (adjust this path as needed)
base_dir = r"C:\Users\simra\Desktop\Extra\CropDiseaseDetection\data\diseases"  # <-- CHANGE THIS

# Paths to original images and labels
images_dir = os.path.join(base_dir, 'images')  # should contain all image files
labels_dir = os.path.join(base_dir, 'labels')  # should contain all .txt label files

# Destination folders
train_images_dir = os.path.join(images_dir, 'train')
val_images_dir = os.path.join(images_dir, 'val')
train_labels_dir = os.path.join(labels_dir, 'train')
val_labels_dir = os.path.join(labels_dir, 'val')

# Create destination folders if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# List of all image files
image_files = [
    f for f in os.listdir(images_dir)
    if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

# Shuffle and split
random.shuffle(image_files)
split_index = int(0.8 * len(image_files))
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Function to move image and its corresponding label
def move_files(file_list, image_dst, label_dst):
    for img_file in file_list:
        # Move image
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(image_dst, img_file)
        print(f"Moving image: {src_img} -> {dst_img}")
        shutil.move(src_img, dst_img)

        # Move label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(label_dst, label_file)

        if os.path.exists(src_label):
            print(f"Moving label: {src_label} -> {dst_label}")
            shutil.move(src_label, dst_label)
        else:
            print(f"⚠️ Warning: Label not found for {img_file} ({src_label})")

    for img_file in file_list:
        # Move image
        src_img = os.path.join(images_dir, img_file)
        dst_img = os.path.join(image_dst, img_file)
        shutil.move(src_img, dst_img)

        # Move label
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(label_dst, label_file)

        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            print(f"⚠️ Warning: Label not found for {img_file}")

# Move training and validation files
move_files(train_files, train_images_dir, train_labels_dir)
move_files(val_files, val_images_dir, val_labels_dir)

print("✅ Dataset split into train/val completed.")
