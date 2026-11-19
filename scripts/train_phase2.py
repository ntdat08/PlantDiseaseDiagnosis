from pathlib import Path
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_NAME = "dpdl-benchmark/plant_village"
CACHE_DIR = "dataset/hf_cache"

MANIFEST_PATH = Path(
    "dataset/processed/split_manifest.csv"
)

MODEL_DIR = Path("models")
EVALUATION_DIR = Path("evaluation/classification")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 38

EPOCHS = 5
UNFREEZE_LAYERS = 20
LEARNING_RATE = 0.00001
RANDOM_STATE = 42

PHASE1_MODEL_PATH = (
    MODEL_DIR / "mobilenetv2_best_phase1.h5"
)

PHASE2_MODEL_PATH = (
    MODEL_DIR / "mobilenetv2_best_phase2.h5"
)

HISTORY_PATH = (
    EVALUATION_DIR / "phase2_history.csv"
)

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print("TensorFlow:", tf.__version__)
print("Bat dau Fine-tuning MobileNetV2 - Phase 2")


# ==========================================
# 2. CHECK PHASE 1 MODEL
# ==========================================

print("\nDang kiem tra Phase 1 model...")

if not PHASE1_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay model: {PHASE1_MODEL_PATH}"
    )

print("Phase 1 model:", PHASE1_MODEL_PATH)


# ==========================================
# 3. LOAD MANIFEST
# ==========================================

print("\nDang doc split manifest...")

manifest = pd.read_csv(MANIFEST_PATH)

train_df = manifest[
    manifest["split"] == "train"
]

val_df = manifest[
    manifest["split"] == "validation"
]

train_records = list(
    train_df[
        ["dataset_index", "label_id"]
    ].itertuples(index=False, name=None)
)

val_records = list(
    val_df[
        ["dataset_index", "label_id"]
    ].itertuples(index=False, name=None)
)

print("Train images:", len(train_records))
print("Validation images:", len(val_records))


# ==========================================
# 4. CALCULATE CLASS WEIGHTS
# ==========================================

print("\nDang tinh class weights...")

train_labels = train_df["label_id"].to_numpy()

classes = np.arange(NUM_CLASSES)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_labels
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("Class weight minimum:", min(class_weights.values()))
print("Class weight maximum:", max(class_weights.values()))


# ==========================================
# 5. LOAD HUGGING FACE DATASET
# ==========================================

print("\nDang tai PlantVillage...")

hf_dataset = load_dataset(
    DATASET_NAME,
    cache_dir=CACHE_DIR
)

data = hf_dataset["train"]


# ==========================================
# 6. CREATE TF.DATA PIPELINE
# ==========================================

def create_tf_dataset(records, training=False):

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

    if training:
        ds = ds.shuffle(
            buffer_size=2048,
            reshuffle_each_iteration=True
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


print("\nDang tao TensorFlow datasets...")

train_ds = create_tf_dataset(
    train_records,
    training=True
)

val_ds = create_tf_dataset(
    val_records,
    training=False
)


# ==========================================
# 7. LOAD BEST PHASE 1 MODEL
# ==========================================

print("\nDang load best Phase 1 model...")

model = tf.keras.models.load_model(
    PHASE1_MODEL_PATH
)

print("Model loaded:", model.name)


# ==========================================
# 8. UNFREEZE LAST LAYERS
# ==========================================

print("\nDang mo khoa cac lop cuoi MobileNetV2...")

# Tim MobileNetV2 backbone trong model
base_model = next(
    layer
    for layer in model.layers
    if isinstance(layer, tf.keras.Model)
)

# Cho phep fine-tuning backbone
base_model.trainable = True

# Dong bang tat ca cac lop truoc
for layer in base_model.layers:
    layer.trainable = False

# Mo khoa 20 lop cuoi, ngoai tru BatchNormalization
for layer in base_model.layers[-UNFREEZE_LAYERS:]:

    if not isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = True

# Thong ke cac lop
trainable_layers = sum(
    1 for layer in base_model.layers
    if layer.trainable
)

frozen_layers = sum(
    1 for layer in base_model.layers
    if not layer.trainable
)

print("Total backbone layers:", len(base_model.layers))
print("Trainable backbone layers:", trainable_layers)
print("Frozen backbone layers:", frozen_layers)


# ==========================================
# 9. RECOMPILE MODEL
# ==========================================

print("\nDang compile model...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

trainable_params = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.trainable_weights
)

print("\nModel trainable parameters:")
print(trainable_params)


# ==========================================
# 10. CALLBACKS
# ==========================================

callbacks = [

    ModelCheckpoint(
        filepath=str(PHASE2_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=1,
        min_lr=1e-7,
        verbose=1
    ),

    CSVLogger(
        filename=str(HISTORY_PATH),
        append=False
    )
]


# ==========================================
# 11. TRAIN PHASE 2
# ==========================================

print("\n========== START PHASE 2 TRAINING ==========")

print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Unfreeze layers:", UNFREEZE_LAYERS)
print("BatchNormalization: Frozen")
print("Class weighting: Enabled")

start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

elapsed = time.time() - start_time


# ==========================================
# 12. SAVE TRAINING SUMMARY
# ==========================================

history_df = pd.DataFrame(history.history)

print("\n========== PHASE 2 TRAINING SUMMARY ==========")

print("Completed epochs:", len(history_df))

print(
    "Best training accuracy:",
    f"{history_df['accuracy'].max():.4f}"
)

print(
    "Best validation accuracy:",
    f"{history_df['val_accuracy'].max():.4f}"
)

print(
    "Best validation loss:",
    f"{history_df['val_loss'].min():.4f}"
)

print(
    "Training time:",
    f"{elapsed / 60:.2f} minutes"
)

print("\n========== SAVED FILES ==========")

print("Best Phase 2 model:", PHASE2_MODEL_PATH)
print("Training history:", HISTORY_PATH)

print("\nPHASE 2 TRAINING COMPLETED!")
