import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import os
import pickle

# Paths
CSV_PATH = 'crop_disease_characteristics.csv'  # Your disease dataset CSV path
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'disease_tabnet_model.zip')
ENCODER_PATH = os.path.join(MODEL_DIR, 'disease_label_encoder.pkl')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(CSV_PATH)

# If your dataset uses Yes/No instead of 1/0, convert them:
# (skip this if data is already 0/1)
df.replace({'Yes': 1, 'No': 0}, inplace=True)

# Split features and label
X = df.drop(columns=['label'])  # Make sure 'label' is your target column
y = df['label']

# Encode labels (in case labels are categorical)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X.values, y_encoded, test_size=0.2, random_state=42
)

# Convert to float32 (TabNet requires float32)
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize TabNet model
model = TabNetClassifier(device_name=device)

# Train model
model.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_val, y_val)],
    eval_name=["val"],
    eval_metric=["accuracy"],
    max_epochs=100,
    patience=10,
    batch_size=2,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Save model
model.save_model(MODEL_PATH)

# Save label encoder
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"✅ Training complete.\n📦 Model saved to: {MODEL_PATH}\n🧠 Label encoder saved to: {ENCODER_PATH}")
