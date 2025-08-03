from ultralytics import YOLO
import time

model = YOLO('yolov8s.pt')

model.train(
    data='data/insects/insect.yaml',
    epochs=30,
    imgsz=640,
    batch=4,  # lower batch size for CPU
    name=f'yolov8_insect_{int(time.time())}',
    workers=1,  # very low CPU usage
    patience=5,  # early stopping
)
