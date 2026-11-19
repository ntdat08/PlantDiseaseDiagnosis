import time
import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# 1. CONFIGURATION
# ==========================================

MANIFEST_PATH = "dataset/processed/split_manifest.csv"
CACHE_DIR = "dataset/hf_cache"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

TRAIN_SAMPLE_SIZE = 512
VAL_SAMPLE_SIZE = 128

print("TensorFlow:", tf.__version__)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nDang doc manifest...")

manifest = pd.read_csv(MANIFEST_PATH)

train_df = manifest[
    manifest["split"] == "train"
]

val_df = manifest[
    manifest["split"] == "validation"
]

train_sample, _ = train_test_split(
    train_df,
    train_size=TRAIN_SAMPLE_SIZE,
    stratify=train_df["label_id"],
    random_state=42
)

val_sample, _ = train_test_split(
    val_df,
    train_size=VAL_SAMPLE_SIZE,
    stratify=val_df["label_id"],
    random_state=42
)

train_records = list(
    train_sample[
        ["dataset_index", "label_id"]
    ].itertuples(index=False, name=None)
)

val_records = list(
    val_sample[
        ["dataset_index", "label_id"]
    ].itertuples(index=False, name=None)
)

print("Train sample:", len(train_records))
print("Validation sample:", len(val_records))


# ==========================================
# 3. LOAD HUGGING FACE DATASET
# ==========================================

print("\nDang tai dataset...")

hf_dataset = load_dataset(
    "dpdl-benchmark/plant_village",
    cache_dir=CACHE_DIR
)

data = hf_dataset["train"]


# ==========================================
# 4. CREATE TF.DATA DATASET
# ==========================================

def create_tf_dataset(records, training=False):

    def generator():

        for index, label in records:

            image = data[int(index)]["image"].convert("RGB")

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
        generator,
        output_signature=output_signature
    )

    if training:
        ds = ds.shuffle(
            buffer_size=256,
            reshuffle_each_iteration=True
        )

    def process(image, label):

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
        process,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = create_tf_dataset(
    train_records,
    training=True
)

val_ds = create_tf_dataset(
    val_records,
    training=False
)


# ==========================================
# 5. BUILD MOBILENETV2 MODEL
# ==========================================

print("\nDang khoi tao MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    38,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 6. BENCHMARK ONE EPOCH
# ==========================================

print("\nBat dau benchmark...")

start_time = time.time()

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    verbose=1
)

elapsed = time.time() - start_time

estimated_epoch = (
    elapsed
    * 37997
    / TRAIN_SAMPLE_SIZE
)

print("\n========== BENCHMARK RESULT ==========")

print(f"Benchmark images: {TRAIN_SAMPLE_SIZE}")
print(f"Elapsed time: {elapsed / 60:.2f} minutes")

print(
    "Estimated full training epoch:",
    f"{estimated_epoch / 60:.2f} hours"
)

print("\nBENCHMARK COMPLETED!")
