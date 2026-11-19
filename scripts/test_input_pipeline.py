from pathlib import Path

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

MANIFEST_PATH = Path(
    "dataset/processed/split_manifest.csv"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

print("TensorFlow version:", tf.__version__)
print("Bat dau kiem tra input pipeline...")


# ==========================================
# 2. LOAD DATASET AND MANIFEST
# ==========================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay: {MANIFEST_PATH}"
    )

print("\nDang doc split manifest...")

manifest = pd.read_csv(MANIFEST_PATH)

print("Dang tai PlantVillage...")

dataset = load_dataset(
    DATASET_NAME,
    cache_dir=CACHE_DIR
)

data = dataset["train"]

print("Tong so anh trong dataset:", len(data))
print("Tong so dong trong manifest:", len(manifest))


# ==========================================
# 3. PREPARE IMAGE RECORDS
# ==========================================

train_records = list(
    manifest[manifest["split"] == "train"][
        ["dataset_index", "label_id"]
    ].itertuples(index=False, name=None)
)

val_records = list(
    manifest[manifest["split"] == "validation"][
        ["dataset_index", "label_id"]
    ].itertuples(index=False, name=None)
)

test_records = list(
    manifest[manifest["split"] == "test"][
        ["dataset_index", "label_id"]
    ].itertuples(index=False, name=None)
)

print("\n========== DATA COUNTS ==========")
print("Train:", len(train_records))
print("Validation:", len(val_records))
print("Test:", len(test_records))


# ==========================================
# 4. CREATE TENSORFLOW DATASET
# ==========================================

def create_tf_dataset(records, training=False):

    def image_generator():

        for dataset_index, label_id in records:

            # Lay anh tu Hugging Face dataset
            item = data[int(dataset_index)]

            image = item["image"].convert("RGB")

            # Chuyen PIL Image thanh NumPy array
            image_array = np.asarray(
                image,
                dtype=np.uint8
            )

            yield image_array, np.int32(label_id)

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

    tf_dataset = tf.data.Dataset.from_generator(
        image_generator,
        output_signature=output_signature
    )

    # Shuffle chi ap dung cho tap Train
    if training:
        tf_dataset = tf_dataset.shuffle(
            buffer_size=2048,
            reshuffle_each_iteration=True
        )

    # Resize va chuan hoa anh
    def preprocess_image(image, label):

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

    tf_dataset = tf_dataset.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    tf_dataset = tf_dataset.batch(
        BATCH_SIZE
    )

    tf_dataset = tf_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return tf_dataset


# ==========================================
# 5. BUILD THREE DATASETS
# ==========================================

print("\nDang tao TensorFlow datasets...")

train_dataset = create_tf_dataset(
    train_records,
    training=True
)

val_dataset = create_tf_dataset(
    val_records,
    training=False
)

test_dataset = create_tf_dataset(
    test_records,
    training=False
)


# ==========================================
# 6. TEST ONE BATCH
# ==========================================

print("\nDang kiem tra batch dau tien...")

images, labels = next(iter(train_dataset))

print("\n========== BATCH INFORMATION ==========")

print("Image shape:", images.shape)
print("Label shape:", labels.shape)

print("Image dtype:", images.dtype)
print("Label dtype:", labels.dtype)

print(
    "Image minimum value:",
    float(tf.reduce_min(images))
)

print(
    "Image maximum value:",
    float(tf.reduce_max(images))
)

print(
    "Label minimum:",
    int(tf.reduce_min(labels))
)

print(
    "Label maximum:",
    int(tf.reduce_max(labels))
)


# ==========================================
# 7. FINAL VERIFICATION
# ==========================================

assert images.shape[1:] == (224, 224, 3)

assert labels.shape[0] <= BATCH_SIZE

assert images.dtype == tf.float32

assert labels.dtype == tf.int32

assert np.isfinite(images.numpy()).all()

assert float(tf.reduce_min(images)) >= -1.01
assert float(tf.reduce_max(images)) <= 1.01

print("\n========== PIPELINE VERIFICATION ==========")
print("Image resizing: PASSED")
print("RGB format: PASSED")
print("MobileNetV2 preprocessing: PASSED")
print("Batch processing: PASSED")
print("Image values in expected range: PASSED")

print("\nINPUT PIPELINE READY!")
