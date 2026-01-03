# skin_tone_test.py
# Run with: python3 skin_tone_test.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "./skin_tone_model.h5"
IMAGE_PATH = "./test.jpg"   # <-- path to image you want to test
IMG_SIZE = (224, 224)

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# =========================
# LOAD & PREPROCESS IMAGE
# =========================
img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# =========================
# PREDICTION
# =========================
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

print("Raw predictions:", predictions)
print("Predicted class index:", predicted_class)
