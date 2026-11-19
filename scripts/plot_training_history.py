import os
import pandas as pd
import matplotlib.pyplot as plt

# Duong dan
HISTORY_PATH = "evaluation/classification/phase1_history.csv"
OUTPUT_DIR = "evaluation/classification"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Doc lich su huan luyen
print("Dang doc training history...")

history = pd.read_csv(HISTORY_PATH)

print("\nCac cot du lieu:")
print(history.columns.tolist())

print("\nTraining history:")
print(history.to_string(index=False))

epochs = range(1, len(history) + 1)

# 1. Accuracy
plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.title("MobileNetV2 - Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(list(epochs))
plt.grid(True)
plt.legend()
plt.tight_layout()

accuracy_path = os.path.join(
    OUTPUT_DIR, "phase1_accuracy.png"
)

plt.savefig(accuracy_path, dpi=300)
plt.show()

# 2. Loss
plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title("MobileNetV2 - Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(list(epochs))
plt.grid(True)
plt.legend()
plt.tight_layout()

loss_path = os.path.join(
    OUTPUT_DIR, "phase1_loss.png"
)

plt.savefig(loss_path, dpi=300)
plt.show()

print("\nDa luu bieu do:")
print(accuracy_path)
print(loss_path)

print("\nPLOT COMPLETED!")
