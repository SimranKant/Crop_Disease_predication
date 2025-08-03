import os
import cv2
import numpy as np
import random

# Paths
image_dir = 'data/diseases/images/train'
label_dir = 'data/diseases/labels/train'
output_image_dir = 'data/diseases/images/train_aug'
output_label_dir = 'data/diseases/labels/train_aug'

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Helper function to apply horizontal flip
def horizontal_flip(img, bboxes):
    img_flipped = cv2.flip(img, 1)
    h, w = img.shape[:2]
    flipped_bboxes = []
    for cls, x, y, bw, bh in bboxes:
        flipped_x = 1 - x
        flipped_bboxes.append([cls, flipped_x, y, bw, bh])
    return img_flipped, flipped_bboxes

# Iterate through all images
for filename in os.listdir(image_dir):
    if not filename.endswith('.jpg'):
        continue

    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[SKIPPED] Could not load image: {filename}")
        continue

    # Load label
    bboxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"[SKIPPED] Invalid label line in {label_path}: {line.strip()}")
                    continue
                try:
                    cls, x, y, bw, bh = map(float, parts[:5])
                    bboxes.append([cls, x, y, bw, bh])
                except ValueError:
                    print(f"[SKIPPED] Invalid float values in {label_path}: {line.strip()}")
                    continue
    except FileNotFoundError:
        print(f"[SKIPPED] Label file not found for {filename}")
        continue

    if not bboxes:
        print(f"[SKIPPED] No valid bounding boxes in {filename}")
        continue

    # Apply horizontal flip
    image_flipped, bboxes_flipped = horizontal_flip(image, bboxes)

    # Save new image
    new_filename = filename.replace('.jpg', '_aug.jpg')
    cv2.imwrite(os.path.join(output_image_dir, new_filename), image_flipped)

    # Save new labels
    new_label_path = os.path.join(output_label_dir, new_filename.replace('.jpg', '.txt'))
    with open(new_label_path, 'w') as f:
        for cls, x, y, bw, bh in bboxes_flipped:
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"[DONE] Augmented and saved: {new_filename}")
