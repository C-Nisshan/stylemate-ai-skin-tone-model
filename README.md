# Skin Tone Classification API

A deep-learning-based Skin Tone Classification API built using TensorFlow and FastAPI.

The API accepts an image upload and returns:
- Predicted skin tone label
- Confidence score
- Confidence distribution across all classes

## Project Overview

A deep-learning-based Skin Tone Classification API built using TensorFlow + FastAPI.

The API accepts an image and returns:
- Predicted skin tone label
- Confidence score
- Confidence distribution across all classes


## Model Architecture

The model is a Convolutional Neural Network (CNN) trained on face and skin images.

### High-Level Summary

- **Input**: 224 × 224 RGB image
- **Preprocessing**:
  - Resize to 224 × 224
  - Normalize pixel values to [-1, 1]
- **CNN Layers**:
  - Conv2D → ReLU → MaxPooling (repeated 3 times)
- **Fully Connected Layers**:
  - Dense (128 units)
  - Dropout (to prevent overfitting)
  - Output Dense layer with softmax activation
- **Output**: Probability distribution over skin tone classes

## Labels Configuration

File: `labels/skin_tone_labels.txt`

Current contents:
```
0 dark
1 light
2 mid-dark
3 mid-light
```

This defines **4 classes**.  

**Important**: The trained model (`skin_tone_model.h5`) must have exactly 4 output neurons to match the number of labels.  
If your dataset previously had only 3 folders/classes, the model must be retrained for 4 classes; otherwise predictions will be incorrect.

The API code supports both label formats (`"0 dark"` or just `"dark"`), so no code changes are needed.

## API Implementation

- Framework: FastAPI
- Model is loaded once at application startup
- Image processing occurs entirely in memory (no disk writes required)
- All outputs are JSON-serializable (NumPy types converted correctly)

### 3. Predict Skin Tone

**URL**: `POST /predict`  
**Content-Type**: multipart/form-data

**Request Body**:

| Field | Type            | Required |
|-------|-----------------|----------|
| file  | Image (jpg/png) | Yes      |

**Example Response**:

```json
{
  "predicted_skin_tone": "mid-light",
  "confidence": 0.8734,
  "all_confidences": {
    "mid-light": 0.8734,
    "light": 0.0812,
    "mid-dark": 0.0321,
    "dark": 0.0133
  }
}
```

## Swagger UI (Interactive API Testing)

After starting the server, visit:

http://127.0.0.1:8000/docs

You can:
- Upload an image
- Execute the `/predict` endpoint
- View live responses

## How to Run the API (Linux + VS Code)

1. **Create Virtual Environment**

```
python3 -m venv venv
source venv/bin/activate
```
2. **Install Dependencies**

```
pip install -r requirements.txt
```
3. **Start the API Server**

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Test in the browser

- Swagger UI:
  ```
   http://127.0.0.1:8000/docs
  ```
- Health check:
  ```
  http://127.0.0.1:8000/health
  ```
