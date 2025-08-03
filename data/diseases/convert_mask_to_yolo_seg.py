import os
import cv2
import numpy as np

# --- CONFIGURATION ---
mask_dir = "SegmentationClass"
output_dir = "labels"
label_map_path = "labelmap.txt"

# --- PREPARE OUTPUT DIRECTORY ---
os.makedirs(output_dir, exist_ok=True)

# --- PARSE labelmap.txt TO CREATE COLOR → CLASS INDEX MAPPING ---
color_to_index = {}
with open(label_map_path, "r") as f:
    lines = f.readlines()
    for idx, line in enumerate(lines[1:]):  # Skip header line
        line = line.strip()
        if not line or ":" not in line:
            continue

        parts = line.split(":")
        if len(parts) < 2 or not parts[1]:
            continue

        label_name = parts[0]
        color = parts[1].split(",")
        if len(color) != 3:
            continue

        try:
            r, g, b = map(int, color)
            color_to_index[(r, g, b)] = idx  # class 0: first valid label (excluding background)
        except:
            continue

# --- CONVERT EACH PNG MASK TO YOLO SEGMENTATION TXT ---
for mask_name in os.listdir(mask_dir):
    if not mask_name.lower().endswith(".png"):
        continue

    mask_path = os.path.join(mask_dir, mask_name)
    mask = cv2.imread(mask_path)

    if mask is None:
        print(f"⚠️ Failed to load mask: {mask_name}")
        continue

    h, w, _ = mask.shape
    label_file = os.path.join(output_dir, os.path.splitext(mask_name)[0] + ".txt")

    with open(label_file, "w") as out_file:
        for color, cls_id in color_to_index.items():
            if cls_id < 0:
                continue  # skip background if it's mapped to -1

            binary_mask = cv2.inRange(mask, np.array(color), np.array(color))
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) < 10:
                    continue  # skip very small/noisy regions

                contour = contour.squeeze()
                if contour.ndim != 2 or contour.shape[0] < 3:
                    continue  # skip invalid polygons

                normalized = []
                for x, y in contour:
                    normalized.extend([x / w, y / h])

                out_file.write(f"{cls_id} " + " ".join(map(str, normalized)) + "\n")

print("✅ All segmentation masks converted to YOLOv8 format and saved in 'labels/' directory.")
