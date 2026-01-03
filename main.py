import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras.models import load_model

# ================== CONFIGURATION ==================
MODEL_PATH = "model/skin_tone_model.keras"  # ← Updated model path
LABELS_PATH = "labels/skin_tone_labels.txt"
os.makedirs("uploads", exist_ok=True)

# Silence TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ================== LOAD MIGRATED MODEL & LABELS ==================
print("Loading migrated Keras 3 model...")
try:
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print("Model failed to load:", str(e))
    raise e

with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ================== FASTAPI APP ==================
app = FastAPI(title="Skin Tone Classifier API", version="1.0")

def preprocess_image(image: Image.Image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32)
    arr = (arr / 127.5) - 1.0  # Normalize to [-1, 1]
    return arr.reshape(1, 224, 224, 3)

@app.get("/")
async def root():
    return {"message": "Skin Tone API is running → go to /docs"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image")

    # Save uploaded file
    with open(f"uploads/{file.filename}", "wb") as out:
        out.write(image_bytes)

    # Run prediction
    try:
        pred = model.predict(preprocess_image(image), verbose=0)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    idx = int(np.argmax(pred))
    raw_label = class_names[idx]
    predicted_class = raw_label.split(" ", 1)[-1] if " " in raw_label else raw_label

    # Build confidence map
    scores = {
        (name.split(" ", 1)[-1] if " " in name else name): float(pred[i])
        for i, name in enumerate(class_names)
    }

    return {
        "predicted_skin_tone": predicted_class.strip(),
        "confidence": round(float(pred[idx]), 4),
        "all_confidences": dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
