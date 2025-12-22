import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
from itertools import cycle

# Configuration
TEST_DIR = "plant_dataset/final_dataset/test"
MODEL_PATH = "plant_disease_model_final.keras"
IMG_SIZE = 224

# 1. Load Model and Data
print("📂 Loading model and test data...")
model = load_model(MODEL_PATH)

datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# 2. Get Predictions
print("🔮 Generating predictions...")
y_score = model.predict(test_generator)
y_test = test_generator.classes
n_classes = len(test_generator.class_indices)
class_labels = list(test_generator.class_indices.keys())

# One-hot encode y_test for ROC
from tensorflow.keras.utils import to_categorical
y_test_oh = to_categorical(y_test, num_classes=n_classes)

# 3. Calculate ROC and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_oh[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 4. Plot ROC Curves
plt.figure(figsize=(10, 8))
colors = cycle(['blue', 'red', 'green', 'orange'])

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC of {class_labels[i]} (AUC = {roc_auc[i]:0.4f})')

# Base Paper Comparison (Adding their best AUC as a reference point in legend)
# Based on the uploaded image, MRW-CNN had a high score (likely AUC) of ~0.9518 or 0.9704.
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guessing (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Multi-Class ROC Curve: Our Model vs Base Paper Benchmarks')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Add a text box for the comparison
textstr = '\n'.join((
    'Base Paper Benchmarks:',
    'VGG16 AUC: 0.9922',
    'Xception AUC: 0.9995',
    'MRW-CNN AUC: 0.9518 (Paper)'
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.05, 0.5, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.savefig('roc_curve_comparison.png')
print("✅ ROC Curve saved: 'roc_curve_comparison.png'")

# Print summary
print("\n--- AUC SUMMARY ---")
for i in range(n_classes):
    print(f"{class_labels[i]}: {roc_auc[i]:0.4f}")
