import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import json

# =======================
# CONFIGURATION
# =======================
MODEL_PATH = "plant_disease_model_final copy 2.keras"
TEST_DIR = "plant_dataset/final_dataset/val"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = 224

def test_random_samples(num_samples=3):
    print(f"Loading resources for verification...")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        return

    # Load Model
    model = load_model(MODEL_PATH)
    
    # Load Classes
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    # Get All Classes
    classes = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    
    print(f"Testing {num_samples} random samples...")
    print("-" * 50)

    for i in range(num_samples):
        # Pick random class and random image
        random_class = random.choice(classes)
        class_path = os.path.join(TEST_DIR, random_class)
        image_name = random.choice(os.listdir(class_path))
        img_path = os.path.join(class_path, image_name)

        # 1. Load and Preprocess
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # 2. Predict
        preds = model.predict(x, verbose=0)
        pred_idx = np.argmax(preds[0])
        confidence = preds[0][pred_idx]
        pred_label = idx_to_class[pred_idx]

        # 3. Output Result
        status = "CORRECT" if pred_label == random_class else "WRONG"
        print(f"Sample {i+1}:")
        print(f"   Actual:    {random_class}")
        print(f"   Predicted: {pred_label} ({confidence:.2%})")
        print(f"   Result:    {status}")
        print("-" * 50)

if __name__ == "__main__":
    test_random_samples(5)
