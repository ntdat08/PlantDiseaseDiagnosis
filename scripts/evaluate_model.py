from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_NAME = "dpdl-benchmark/plant_village"
CACHE_DIR = "dataset/hf_cache"

MANIFEST_PATH = Path(
    "dataset/processed/split_manifest.csv"
)

MODEL_PATH = Path(
    "models/mobilenetv2_best_phase2.h5"
)

OUTPUT_DIR = Path("evaluation/classification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 38


# Mapping 38 classes
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

print("========== MODEL EVALUATION ==========")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay model: {MODEL_PATH}"
    )

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay manifest: {MANIFEST_PATH}"
    )


# ==========================================
# 3. LOAD TEST MANIFEST
# ==========================================

print("\nDang doc Test manifest...")

manifest = pd.read_csv(MANIFEST_PATH)

test_df = manifest[
    manifest["split"] == "test"
].copy()

test_records = list(
    test_df[
        ["dataset_index", "label_id"]
    ].itertuples(index=False, name=None)
)

y_true = test_df["label_id"].to_numpy(dtype=np.int32)

print("Test images:", len(test_records))
print("Number of classes:", len(np.unique(y_true)))


# ==========================================
# 4. LOAD DATASET
# ==========================================

print("\nDang tai PlantVillage...")

hf_dataset = load_dataset(
    DATASET_NAME,
    cache_dir=CACHE_DIR
)

data = hf_dataset["train"]


# ==========================================
# 5. CREATE TEST PIPELINE
# ==========================================

def create_test_dataset(records):

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

    ds = tf.data.Dataset.from_generator(
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

    ds = ds.map(
        process_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


print("\nDang tao Test dataset...")

test_ds = create_test_dataset(test_records)


# ==========================================
# 6. LOAD BEST MODEL
# ==========================================

print("\nDang load MobileNetV2 Phase 2...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ==========================================
# 7. PREDICTION
# ==========================================

print("\n========== START TEST PREDICTION ==========")

start_time = time.time()

predictions = model.predict(
    test_ds,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

prediction_time = time.time() - start_time

print("\nPrediction completed!")
print("Prediction time:", round(prediction_time / 60, 2), "minutes")


# ==========================================
# 8. CALCULATE METRICS
# ==========================================

print("\n========== TEST METRICS ==========")

accuracy = accuracy_score(y_true, y_pred)

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print(f"Test Accuracy:       {accuracy:.4f}")
print(f"Macro Precision:     {macro_precision:.4f}")
print(f"Macro Recall:        {macro_recall:.4f}")
print(f"Macro F1-score:      {macro_f1:.4f}")
print(f"Weighted Precision:  {weighted_precision:.4f}")
print(f"Weighted Recall:     {weighted_recall:.4f}")
print(f"Weighted F1-score:   {weighted_f1:.4f}")


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

print(report_df.to_string())

report_path = OUTPUT_DIR / "classification_report.csv"

report_df.to_csv(
    report_path,
    encoding="utf-8-sig"
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

print("\nDang tao Confusion Matrix...")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

cm_path = OUTPUT_DIR / "confusion_matrix.csv"

cm_df.to_csv(
    cm_path,
    encoding="utf-8-sig"
)


# Normalize by true class for visualization
row_sums = cm.sum(axis=1, keepdims=True)

cm_normalized = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(cm, dtype=float),
    where=row_sums != 0
)

plt.figure(figsize=(22, 19))

plt.imshow(
    cm_normalized,
    interpolation="nearest",
    cmap="Blues",
    vmin=0,
    vmax=1
)

plt.title("MobileNetV2 Phase 2 - Normalized Confusion Matrix")
plt.xlabel("Predicted Class ID")
plt.ylabel("True Class ID")

plt.xticks(
    np.arange(NUM_CLASSES),
    np.arange(NUM_CLASSES),
    rotation=90
)

plt.yticks(
    np.arange(NUM_CLASSES),
    np.arange(NUM_CLASSES)
)

plt.colorbar(label="Proportion")
plt.tight_layout()

cm_image_path = OUTPUT_DIR / "confusion_matrix.png"

plt.savefig(
    cm_image_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 11. SAVE SUMMARY
# ==========================================

summary = {
    "model": "MobileNetV2 Phase 2",
    "test_images": int(len(y_true)),
    "accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),
    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "weighted_f1": float(weighted_f1),
    "prediction_time_minutes": float(prediction_time / 60)
}

summary_path = OUTPUT_DIR / "test_metrics.json"

with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False
    )


# ==========================================
# 12. FINAL RESULTS
# ==========================================

print("\n========== EVALUATION SUMMARY ==========")

for key, value in summary.items():
    print(f"{key}: {value}")

print("\n========== SAVED FILES ==========")

print("Classification report:", report_path)
print("Confusion matrix CSV:", cm_path)
print("Confusion matrix image:", cm_image_path)
print("Metrics summary:", summary_path)

print("\nMODEL EVALUATION COMPLETED!")
