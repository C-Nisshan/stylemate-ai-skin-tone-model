# skin_tone_model.py
# Run with: python3 skin_tone_model.py

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# CONFIGURATION
# =========================
DATASET_DIR = "./dataset"        # <-- change to your dataset path
MODEL_OUTPUT = "./skin_tone_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

print("Current working directory:", os.getcwd())

# =========================
# DATA GENERATOR
# =========================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = train_generator.num_classes
print("Number of classes:", num_classes)

# =========================
# MODEL
# =========================
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAINING
# =========================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# =========================
# SAVE MODEL
# =========================
model.save(MODEL_OUTPUT)
print(f"Model saved to: {MODEL_OUTPUT}")
