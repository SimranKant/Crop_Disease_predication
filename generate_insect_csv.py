import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# List of 30 binary questions (tomato disease)
questions = [
    "Is there a yellow halo around the spots?",
    "Are the leaf spots circular with concentric rings?",
    "Does the disease begin on the lower leaves?",
    "Are the lesions expanding over time?",
    "Is the center of the spot dry and brown?",
    "Are multiple spots merging to form large blotches?",
    "Does the leaf show signs of early yellowing?",
    "Are stems or fruits also affected?",
    "Are the affected leaves wilting?",
    "Is the infection spreading upward on the plant?",
    "Are concentric rings visible clearly on the leaves?",
    "Is there any rotting seen on fruit?",
    "Are the leaf margins turning brown?",
    "Is the plant under moisture stress?",
    "Is the disease more active during rainy days?",
    "Are nearby tomato plants also showing similar symptoms?",
    "Is there any black moldy growth on the lesion?",
    "Does the disease affect the whole plant?",
    "Is the spot size more than 5mm in diameter?",
    "Are the lesions visible on both sides of the leaf?",
    "Is the infection found only on mature leaves?",
    "Are the leaf veins visible through the lesion?",
    "Is the damage uniform across the field?",
    "Was there previous history of Early Blight in this field?",
    "Is the farmer using resistant tomato varieties?",
    "Was any fungicide recently applied?",
    "Was there poor air circulation in the field?",
    "Was the field irrigated from overhead sprinklers?",
    "Are pruning and sanitation practices followed?",
    "Is there any other crop in the field showing similar spots?"
]

# Generate synthetic binary responses and a label
def generate_data(n_samples=1000):
    data = np.random.randint(0, 2, size=(n_samples, len(questions)))
    
    # Simple labeling rule:
    # If more than 15 "Yes" answers, label=1 (disease present), else 0
    labels = (data.sum(axis=1) > 15).astype(int)

    df = pd.DataFrame(data, columns=questions)
    df['label'] = labels
    return df

# Generate 2000 samples and save to CSV
df = generate_data(2000)
df.to_csv("crop_disease_characteristics.csv", index=False)
print("✅ Synthetic tomato disease data saved to synthetic_tomato_disease_tabnet_data.csv")
