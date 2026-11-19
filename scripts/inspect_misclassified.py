from pathlib import Path
import re

import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import load_dataset
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_NAME = "dpdl-benchmark/plant_village"
CACHE_DIR = "dataset/hf_cache"

MANIFEST_PATH = Path("dataset/processed/split_manifest.csv")
MODEL_PATH = Path("models/mobilenetv2_best_phase2.h5")

OUTPUT_DIR = Path("evaluation/classification/misclassified")
IMAGE_DIR = OUTPUT_DIR / "images"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

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
# 2. CHECK FILES
# ==========================================

print("========== INSPECT MISCLASSIFIED IMAGES ==========")

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(MANIFEST_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 3. LOAD TEST MANIFEST
# ==========================================

print("\nLoading test manifest...")

manifest = pd.read_csv(MANIFEST_PATH)

test_df = manifest[
    manifest["split"] == "test"
].copy().reset_index(drop=True)

records = list(
    test_df[
        ["dataset_index", "label_id"]
    ].itertuples(index=False, name=None)
)

y_true = test_df["label_id"].to_numpy(dtype=np.int32)

print("Test images:", len(records))


# ==========================================
# 4. LOAD DATASET
# ==========================================

print("\nLoading PlantVillage dataset...")

hf_dataset = load_dataset(
    DATASET_NAME,
    cache_dir=CACHE_DIR
)

data = hf_dataset["train"]


# ==========================================
# 5. CREATE TEST PIPELINE
# ==========================================

def image_generator():

    for index, label in records:

        item = data[int(index)]

        image = item["image"].convert("RGB")

        image_array = np.asarray(
            image,
            dtype=np.uint8
        )

        yield image_array, np.int32(label)


output_signature = (
    tf.TensorSpec(
        shape=(None, None, 3),
        dtype=tf.uint8
    ),
    tf.TensorSpec(
        shape=(),
        dtype=tf.int32
    )
)

test_ds = tf.data.Dataset.from_generator(
    image_generator,
    output_signature=output_signature
)


def process_image(image, label):

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = preprocess_input(image)

    return image, label


test_ds = test_ds.map(
    process_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


# ==========================================
# 6. LOAD MODEL
# ==========================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# 7. PREDICTION
# ==========================================

print("\nPredicting test images...")

probabilities = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(
    probabilities,
    axis=1
)

if len(y_pred) != len(y_true):
    raise ValueError(
        f"Prediction count {len(y_pred)} "
        f"does not match test count {len(y_true)}"
    )

wrong_indices = np.where(y_true != y_pred)[0]

print("\nTotal test images:", len(y_true))
print("Correct predictions:", len(y_true) - len(wrong_indices))
print("Misclassified images:", len(wrong_indices))


# ==========================================
# 8. SAVE MISCLASSIFIED IMAGES
# ==========================================

print("\nSaving misclassified images...")

results = []

for position, test_index in enumerate(wrong_indices, start=1):

    row = test_df.iloc[test_index]

    dataset_index = int(row["dataset_index"])
    true_id = int(y_true[test_index])
    pred_id = int(y_pred[test_index])

    item = data[dataset_index]
    image = item["image"].convert("RGB")

    image_filename = (
        f"{position:04d}_"
        f"true-{true_id}_"
        f"pred-{pred_id}.jpg"
    )

    image_path = IMAGE_DIR / image_filename

    image.save(image_path, quality=95)

    confidence = float(probabilities[test_index][pred_id])

    top3_ids = np.argsort(
        probabilities[test_index]
    )[-3:][::-1]

    top3_text = " | ".join(
        f"{CLASS_NAMES[i]}: {probabilities[test_index][i]:.4f}"
        for i in top3_ids
    )

    results.append({
        "dataset_index": dataset_index,
        "original_filename": str(row["filename"]),
        "image_file": image_filename,
        "true_label_id": true_id,
        "actual_class": CLASS_NAMES[true_id],
        "predicted_label_id": pred_id,
        "predicted_class": CLASS_NAMES[pred_id],
        "prediction_confidence": confidence,
        "top3_predictions": top3_text
    })

    if position % 50 == 0:
        print(f"Saved {position}/{len(wrong_indices)} images")


# ==========================================
# 9. EXPORT CSV
# ==========================================

result_df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "misclassified_images.csv"

result_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 10. ERROR SUMMARY
# ==========================================

summary = (
    result_df.groupby(
        ["actual_class", "predicted_class"]
    )
    .size()
    .reset_index(name="wrong_count")
    .sort_values("wrong_count", ascending=False)
)

summary_path = OUTPUT_DIR / "misclassification_summary.csv"

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 11. FINAL RESULTS
# ==========================================

print("\n========== INSPECTION COMPLETED ==========")

print("Test images:", len(y_true))
print("Misclassified images:", len(wrong_indices))
print("Saved image folder:", IMAGE_DIR)
print("Detailed CSV:", csv_path)
print("Error summary:", summary_path)

print("\n========== TOP 10 ERRORS ==========")
print(summary.head(10).to_string(index=False))
