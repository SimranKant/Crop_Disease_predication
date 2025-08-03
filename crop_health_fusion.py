import os
import torch
import pickle
import pandas as pd
import numpy as np
from ultralytics import YOLO
from pytorch_tabnet.tab_model import TabNetClassifier

# ----------------- MODEL PATHS -----------------
INSECT_YOLO_PATH = 'yolov8s.pt'
DISEASE_YOLO_PATH = 'yolov8s-seg.pt'
INSECT_TABNET_PATH = 'models/insect_tabnet_model.zip.zip'
DISEASE_TABNET_PATH = 'models/disease_tabnet_model.zip.zip'
DISEASE_ENCODER_PATH = 'models/disease_label_encoder.pkl'
INSECT_ENCODER_PATH = 'models/label_encoder.pkl'

# ----------------- LOAD MODELS -----------------
print("🔁 Loading models...")
insect_yolo_model = YOLO(INSECT_YOLO_PATH)
disease_yolo_model = YOLO(DISEASE_YOLO_PATH)

insect_tabnet = TabNetClassifier()
insect_tabnet.load_model(INSECT_TABNET_PATH)

disease_tabnet = TabNetClassifier()
disease_tabnet.load_model(DISEASE_TABNET_PATH)

with open(INSECT_ENCODER_PATH, 'rb') as f:
    insect_encoder = pickle.load(f)
with open(DISEASE_ENCODER_PATH, 'rb') as f:
    disease_encoder = pickle.load(f)

# ----------------- GET USER INPUT -----------------
def get_image_path(prompt_text):
    while True:
        path = input(f"{prompt_text}: ").strip('"')
        if os.path.exists(path) and path.lower().endswith(('.jpg', '.jpeg', '.png')):
            return path
        print("❌ Invalid path or file format. Please try again.")

def get_tabular_answers(question_list):
    answers = {}
    print("\nPlease answer the following questions (yes/no):")
    for q in question_list:
        while True:
            ans = input(f"{q}: ").strip().lower()
            if ans in ['yes', 'no', 'y', 'n']:
                answers[q] = 1 if ans in ['yes', 'y'] else 0
                break
            else:
                print("❌ Invalid input. Enter 'yes' or 'no'.")
    return answers

# ----------------- MODEL FUNCTIONS -----------------
def run_yolo_detection(model, image_path):
    results = model(image_path, verbose=False)
    return len(results[0].boxes) > 0

def run_tabnet_prediction(model, label_encoder, input_data):
    input_df = pd.DataFrame([input_data])
    pred_idx = model.predict(input_df.values)[0]
    decoded_label = label_encoder.inverse_transform([int(pred_idx)])[0]
    return str(decoded_label).strip().lower() not in ['no disease', 'none', 'no insect']

# ----------------- DYNAMIC INPUT -----------------
disease_questions = [
    "Are there yellow or brown spots on the leaves?",
    "Are the leaf edges curled or dry?",
    "Is there powdery substance on the leaf surface?",
    "Are there black or brown lesions on the stem?",
    "Is the plant wilting even with adequate water?",
    "Is the fruit discolored or rotting?",
    "Are leaves falling off prematurely?",
    "Are veins on leaves turning yellow?",
    "Is there white or grey mold visible?",
    "Are there irregular patterns on leaves?",
    "Is there leaf blight or scorch present?",
    "Are stems or roots soft and discolored?",
    "Are there signs of cankers or galls?",
    "Are flowers stunted or deformed?",
    "Is the growth of the plant stunted?",
    "Are there sticky substances (honeydew) on the leaves?",
    "Are there signs of bacterial ooze?",
    "Is the plant's yield lower than expected?",
    "Are there black dots or fungal spores visible?",
    "Are the leaf tips turning necrotic (dead)?",
    "Do the leaves show mosaic or mottling patterns?",
    "Are roots knotted or deformed?",
    "Are leaves showing interveinal chlorosis?",
    "Is there abnormal branching or bushiness?",
    "Are stems cracked or peeling?",
    "Is the leaf surface rough or blistered?",
    "Are lesions circular or ringed in shape?",
    "Are seeds or pods discolored or shriveled?",
    "Are there concentric rings on the fruit?",
    "Do lesions expand rapidly in wet conditions?"
]

insect_questions = [
    "Are there visible insects on the leaves?",
    "Are leaves being chewed or eaten?",
    "Is there leaf mining (white or clear squiggly trails)?",
    "Are there small black or green insects (aphids)?",
    "Is there sticky honeydew on the leaves?",
    "Are there caterpillars or larvae on the plant?",
    "Are there beetles present on the stem or leaves?",
    "Are the leaves curling due to insect activity?",
    "Is there silk or webbing (e.g. spider mites)?",
    "Are the roots damaged or hollow?",
    "Is the stem bored or tunneled?",
    "Are fruits punctured or damaged?",
    "Are there whiteflies or flying pests?",
    "Are there ants farming insects on the plant?",
    "Are leaves rolled or folded unnaturally?",
    "Are there dark excrements or frass on leaves?",
    "Are seeds or pods attacked by pests?",
    "Is the plant's top growth wilting suddenly?",
    "Are insect eggs visible under leaves?",
    "Are stems bent or weak at joints?",
    "Do leaves appear skeletonized (only veins remain)?",
    "Is there gumming or resin from insect feeding?",
    "Are buds or flowers missing or malformed?",
    "Are pests visible at night using a torch?",
    "Are insects jumping or flying when disturbed?",
    "Are leaves turning yellow suddenly?",
    "Are there red spider mites or rust-colored dots?",
    "Are insects attacking only young parts of the plant?",
    "Are tunnels visible in fruits or pods?",
    "Is there dead heart in young plants (center shoot dead)?"
]


print("\n📸 Provide image paths:")
input_disease_image = get_image_path("Path to disease image")
input_insect_image = get_image_path("Path to insect image")

input_disease_tabular = get_tabular_answers(disease_questions)
input_insect_tabular = get_tabular_answers(insect_questions)

# ----------------- RUN INFERENCE -----------------
output_1 = run_yolo_detection(disease_yolo_model, input_disease_image)
output_2 = run_yolo_detection(insect_yolo_model, input_insect_image)
output_3 = run_tabnet_prediction(disease_tabnet, disease_encoder, input_disease_tabular)
output_4 = run_tabnet_prediction(insect_tabnet, insect_encoder, input_insect_tabular)

final_output = {
    "Crop Disease Present": output_1 or output_3,
    "Crop Insect Present": output_2 or output_4
}

# ----------------- DISPLAY OUTPUT -----------------
print("\n🧠 Multimodal Inference Results")
print("--------------------------------")
print(f"📸 YOLO - Disease detected: {output_1}")
print(f"📸 YOLO - Insect detected:  {output_2}")
print(f"📊 TabNet - Disease detected: {output_3}")
print(f"📊 TabNet - Insect detected:  {output_4}")
print("\n✅ Final Output:")
print(final_output)
