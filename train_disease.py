from ultralytics import YOLO

# Load the YOLOv8 segmentation model
model = YOLO("yolov8s-seg.pt")  # You can use yolov8n-seg.pt, yolov8m-seg.pt, etc. if needed

# Train the model
model.train(
    data="data/diseases/disease.yaml",  # path to your dataset YAML
    epochs=50,
    imgsz=640,
    batch=2,
    name="yolov8s-seg-disease",
    project="runs/disease_training",
    task="segment"
)
