import os
import sys
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
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = 224

def predict_single_image(img_path):
    # 1. Check Paths
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    if not os.path.exists(CLASS_INDICES_PATH):
        print(f"Error: Class indices file '{CLASS_INDICES_PATH}' not found.")
        return

    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found.")
        return

    # 2. Load Resources
    print("Loading model and class indices...")
    model = load_model(MODEL_PATH)
    
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    # 3. Preprocess Image
    print(f"Processing image: {os.path.basename(img_path)}")
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # 4. Predict
    print("Running prediction...")
    preds = model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    label = idx_to_class[class_idx]

    # 5. Output Result
    print("\n" + "="*40)
    print("         PREDICTION RESULT")
    print("="*40)
    print(f"Class:      {label}")
    print(f"Confidence: {confidence:.2%}")
    print("="*40)

    # Note: For maize southern rust specifically, the model is ~80.78% accurate
    # For healthy rice leaf, it is ~90.66% accurate

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_single_image(sys.argv[1])
    else:
        print("\nUsage: python predict.py <path_to_image>")
        print("Example: python predict.py my_leaf.png")
        
        # Interactive mode if no argument provided
        target = input("\nEnter image path: ").strip('"')
        if target:
            predict_single_image(target)
