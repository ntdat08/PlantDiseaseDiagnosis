from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = Path("models/mobilenetv2_best_phase2.h5")

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
# 3. FORMAT CLASS NAME
# ==========================================

def format_class_name(class_name):

    plant, disease = class_name.split("___", 1)

    plant = plant.replace("_", " ")
    disease = disease.replace("_", " ").strip()

    return plant, disease


# ==========================================
# 4. PREPROCESS IMAGE
# ==========================================

def preprocess_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay anh: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    original_size = image.size

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = preprocess_input(image_array)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array, original_size


# ==========================================
# 5. LOAD MODEL
# ==========================================

def load_trained_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Khong tim thay model: {MODEL_PATH}"
        )

    print("\nDang tai MobileNetV2 Phase 2...")

    model = load_model(MODEL_PATH)

    print("Model loaded successfully!")

    return model


# ==========================================
# 6. PREDICT IMAGE
# ==========================================

def predict_image(model, image_path):

    image_array, original_size = preprocess_image(
        image_path
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    if len(predictions) != NUM_CLASSES:
        raise ValueError(
            f"Model tra ve {len(predictions)} lop, "
            f"nhung mong doi {NUM_CLASSES} lop."
        )

    top_indices = np.argsort(predictions)[::-1][:TOP_K]

    print("\n========== PREDICTION RESULT ==========")

    print("Image:", image_path)
    print("Original size:", original_size)

    for rank, class_id in enumerate(top_indices, start=1):

        class_name = CLASS_NAMES[class_id]

        plant, disease = format_class_name(class_name)

        confidence = float(predictions[class_id]) * 100

        print(f"\nTop {rank}")
        print("Plant:", plant)
        print("Prediction:", disease)
        print(f"Confidence: {confidence:.2f}%")

    best_class_id = int(top_indices[0])

    best_confidence = float(
        predictions[best_class_id]
    ) * 100

    print("\n=======================================")

    return {
        "class_id": best_class_id,
        "class_name": CLASS_NAMES[best_class_id],
        "confidence": best_confidence
    }


# ==========================================
# 7. MAIN
# ==========================================

def main():

    print("PLANT DISEASE IMAGE CLASSIFICATION")
    print("Model: MobileNetV2 Phase 2")

    if len(sys.argv) >= 2:
        image_path = Path(sys.argv[1])
    else:
        image_path = Path(
            input("\nNhap duong dan anh la cay: ").strip().strip('"')
        )

    try:

        model = load_trained_model()

        result = predict_image(
            model,
            image_path
        )

        print("\n========== FINAL RESULT ==========")
        print("Class ID:", result["class_id"])
        print("Class:", result["class_name"])
        print(
            f"Confidence: {result['confidence']:.2f}%"
        )

    except Exception as error:

        print("\nERROR:", error)


if __name__ == "__main__":
    main()
