import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


# ==========================================
# 1. CONFIGURATION
# ==========================================

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 38

print("TensorFlow version:", tf.__version__)


# ==========================================
# 2. LOAD MOBILENETV2
# ==========================================

print("\nDang tai MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Dong bang cac layer backbone
base_model.trainable = False

print("Da tai MobileNetV2 thanh cong!")


# ==========================================
# 3. BUILD CLASSIFICATION MODEL
# ==========================================

inputs = layers.Input(
    shape=(224, 224, 3),
    name="input_image"
)

# Trich xuat dac trung tu anh
x = base_model(
    inputs,
    training=False
)

# Giam chieu dac trung
x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

# Giam overfitting
x = layers.Dropout(
    0.3,
    name="dropout"
)(x)

# Phan loai 38 lop
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="classification"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="PlantDisease_MobileNetV2"
)


# ==========================================
# 4. COMPILE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 5. DISPLAY MODEL INFORMATION
# ==========================================

print("\n========== MODEL SUMMARY ==========")

model.summary()

total_params = model.count_params()

trainable_params = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.trainable_weights
)

print("\nTotal parameters:", total_params)
print("Trainable parameters:", trainable_params)


# ==========================================
# 6. TEST FORWARD PASS
# ==========================================

print("\nDang kiem tra model...")

dummy_images = tf.random.uniform(
    shape=(2, 224, 224, 3),
    minval=-1.0,
    maxval=1.0
)

predictions = model(
    dummy_images,
    training=False
)

print("Prediction shape:", predictions.shape)

print(
    "Probability sum:",
    tf.reduce_sum(predictions, axis=1).numpy()
)

assert predictions.shape == (2, NUM_CLASSES)

assert tf.reduce_all(
    tf.abs(
        tf.reduce_sum(predictions, axis=1) - 1.0
    ) < 1e-5
)

print("\n========== MODEL VERIFICATION ==========")
print("MobileNetV2 backbone: PASSED")
print("Classification layer: PASSED")
print("Output classes: 38")
print("Softmax probability: PASSED")

print("\nMODEL READY FOR TRAINING!")
