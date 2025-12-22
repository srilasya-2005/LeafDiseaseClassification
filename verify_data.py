
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

BASE_DIR = "plant_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "final_dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "final_dataset", "val")
IMG_SIZE = 224
BATCH_SIZE = 8

def verify():
    print(f"Checking directories...")
    print(f"Train: {os.path.exists(TRAIN_DIR)}")
    print(f"Val:   {os.path.exists(VAL_DIR)}")

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    print("Flowing from train...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    print("Flowing from val...")
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    print(f"Train Classes: {train_generator.num_classes}")
    print(f"Val Classes:   {val_generator.num_classes}")
    print(f"Class Indices: {train_generator.class_indices}")

    x, y = next(train_generator)
    print(f"Train Batch X: {x.shape}")
    print(f"Train Batch Y: {y.shape}")

    x, y = next(val_generator)
    print(f"Val Batch X: {x.shape}")
    print(f"Val Batch Y: {y.shape}")

if __name__ == "__main__":
    verify()
