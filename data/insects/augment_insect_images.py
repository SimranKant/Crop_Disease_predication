import os
import cv2
import albumentations as A
import numpy as np

# Paths
BASE_DIR = 'data/insects/images/train'
LABEL_DIR = 'data/insects/labels/train'
AUG_IMAGE_DIR = 'data/insects/images/train_aug'
AUG_LABEL_DIR = 'data/insects/labels/train_aug'

os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
os.makedirs(AUG_LABEL_DIR, exist_ok=True)

# Albumentations augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.ColorJitter(p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

aug_count = 0
for filename in os.listdir(BASE_DIR):
    if not filename.endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(BASE_DIR, filename)
    label_path = os.path.join(LABEL_DIR, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

    if not os.path.exists(label_path):
        print(f"Skipping {filename}, no label found.")
        continue

    image = cv2.imread(image_path)
    h, w, _ = image.shape

    with open(label_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []
    for line in lines:
        cls, x_center, y_center, width, height = map(float, line.strip().split())
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(int(cls))

    # Apply augmentation
    try:
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        if len(aug_bboxes) == 0:
            print(f"Skipping {filename} - no valid bboxes after augmentation")
            continue

        # Save augmented image and label
        aug_image_name = filename.replace('.jpg', '_aug.jpg').replace('.png', '_aug.png')
        aug_label_name = aug_image_name.replace('.jpg', '.txt').replace('.png', '.txt')

        cv2.imwrite(os.path.join(AUG_IMAGE_DIR, aug_image_name), aug_image)

        with open(os.path.join(AUG_LABEL_DIR, aug_label_name), 'w') as f:
            for bbox, cls in zip(aug_bboxes, aug_labels):
                x_c, y_c, w_b, h_b = bbox
                f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w_b:.6f} {h_b:.6f}\n")

        aug_count += 1

    except Exception as e:
        print(f"Error augmenting {filename}: {e}")

print(f"✅ Augmentation completed and saved. Total augmented images: {aug_count}")
