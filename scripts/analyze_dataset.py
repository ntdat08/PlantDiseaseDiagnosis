from datasets import load_dataset
from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from class_names import CLASS_NAMES

print("Dang phan tich PlantVillage...")

dataset = load_dataset(
    "dpdl-benchmark/plant_village",
    cache_dir="dataset/hf_cache"
)

data = dataset["train"]
counts = Counter(data["label"])

# Kiem tra tinh hop le cua nhan
actual_labels = set(counts.keys())
expected_labels = set(range(len(CLASS_NAMES)))

if actual_labels != expected_labels:
    raise ValueError("Nhãn dữ liệu không khớp với danh sách 38 lớp!")

# Tao bang thong ke
results = []

for label_id, class_name in enumerate(CLASS_NAMES):
    results.append({
        "label_id": label_id,
        "class_name": class_name,
        "image_count": counts[label_id]
    })

df = pd.DataFrame(results)

# Tinh ti le phan tram
df["percentage"] = (
    df["image_count"] / len(data) * 100
).round(2)

# Sap xep theo so luong anh
df_sorted = df.sort_values(
    by="image_count",
    ascending=False
)

print("\n========== DATASET SUMMARY ==========")
print("Total images:", len(data))
print("Total classes:", len(CLASS_NAMES))
print("Largest class:", df_sorted.iloc[0]["class_name"])
print("Smallest class:", df_sorted.iloc[-1]["class_name"])

print("\n========== CLASS DISTRIBUTION ==========")
print(df_sorted.to_string(index=False))

# Tao thu muc ket qua
output_dir = Path("evaluation/classification")
output_dir.mkdir(parents=True, exist_ok=True)

# Luu CSV
df.to_csv(
    output_dir / "class_distribution.csv",
    index=False,
    encoding="utf-8-sig"
)

# Ve bieu do
plt.figure(figsize=(16, 9))

plt.barh(
    df_sorted["class_name"],
    df_sorted["image_count"]
)

plt.xlabel("Number of Images")
plt.ylabel("Plant Disease Classes")
plt.title("PlantVillage - Class Distribution")
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    output_dir / "class_distribution.png",
    dpi=200
)

plt.close()

print("\nDa luu ket qua tai:", output_dir)
print("HOAN THANH!")
