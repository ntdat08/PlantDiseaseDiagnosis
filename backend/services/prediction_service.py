from pathlib import Path
from io import BytesIO

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "mobilenetv2_best_phase2.h5"
)

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 38
TOP_K = 3


# ==========================================
# 2. CLASS NAMES
# ==========================================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# ==========================================
# 3. LOAD MODEL
# ==========================================

model = None


def load_prediction_model():

    global model

    if model is not None:
        return model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("Loading MobileNetV2 Phase 2...")

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    if model.output_shape[-1] != NUM_CLASSES:
        raise ValueError(
            "Model output does not match 38 classes."
        )

    print("MobileNetV2 loaded successfully!")

    return model


# ==========================================
# 4. FORMAT CLASS NAME
# ==========================================

def format_class_name(class_name):

    plant, disease = class_name.split("___", 1)

    plant = plant.replace("_", " ").replace(",", "")
    disease = disease.replace("_", " ").strip()

    return plant, disease


# ==========================================
# 5. PREPROCESS IMAGE
# ==========================================

def preprocess_image(image_bytes):

    try:

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):

        raise ValueError(
            "Invalid image file. Please upload a valid image."
        )

    original_width, original_height = image.size

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = preprocess_input(
        image_array
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array, original_width, original_height


# ==========================================
# 6. PREDICT IMAGE
# ==========================================

def predict_image(image_bytes):

    trained_model = load_prediction_model()

    image_array, width, height = preprocess_image(
        image_bytes
    )

    predictions = trained_model.predict(
        image_array,
        verbose=0
    )[0]

    if len(predictions) != NUM_CLASSES:
        raise ValueError(
            "Unexpected number of model predictions."
        )

    top_indices = np.argsort(
        predictions
    )[::-1][:TOP_K]

    results = []

    for rank, class_id in enumerate(top_indices, start=1):

        class_id = int(class_id)

        class_name = CLASS_NAMES[class_id]

        plant, disease = format_class_name(
            class_name
        )

        confidence = float(
            predictions[class_id]
        ) * 100

        results.append({
            "rank": rank,
            "class_id": class_id,
            "class_name": class_name,
            "plant": plant,
            "disease": disease,
            "confidence": round(confidence, 2)
        })

    return {
        "image": {
            "width": width,
            "height": height
        },
        "top_prediction": results[0],
        "top_predictions": results,
        "model": "MobileNetV2 Phase 2"
    }
