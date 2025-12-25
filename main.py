# main.py
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io

# ================== CONFIGURATION ==================
MODEL_PATH = "model/skin_tone_model.h5"
LABELS_PATH = "labels/skin_tone_labels.txt"
os.makedirs("uploads", exist_ok=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide TF warnings

# ================== LOAD MODEL & LABELS ==================
print("Loading skin tone model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ================== FASTAPI APP ==================
app = FastAPI(title="Skin Tone Classifier API", version="1.0")

def preprocess_image(image: Image.Image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

@app.get("/")
async def root():
    return {"message": "Skin Tone API is running → go to /docs"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image")

    # Predict
    data = preprocess_image(image)
    prediction = model.predict(data, verbose=0)[0]  # shape = (num_classes,)
    index = int(np.argmax(prediction))
    confidence = float(prediction[index])               # ← convert to Python float
    raw_label = class_names[index]

    # Clean label (handles both "0 light" and "light" formats)
    predicted_class = raw_label.split(" ", 1)[-1].strip() if " " in raw_label else raw_label.strip()

    # Convert ALL numpy floats → Python float for JSON
    all_scores = {
        (name.split(" ", 1)[-1].strip() if " " in name else name.strip()): float(prediction[i])
        for i, name in enumerate(class_names)
    }

    return {
        "predicted_skin_tone": predicted_class,
        "confidence": round(confidence, 4),
        "all_confidences": dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}