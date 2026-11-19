from datasets import load_dataset
from collections import Counter
from pathlib import Path
from PIL import Image
import csv

DATASET_ID = "dpdl-benchmark/plant_village"

print("Dang doc dataset tu cache...")

dataset = load_dataset(
    DATASET_ID,
    cache_dir="dataset/hf_cache"
)

data = dataset["train"]

print("\nTong so anh:", len(data))
print("Cac cot:", data.column_names)

# Thu muc luu anh mau
preview_dir = Path("dataset/preview_by_label")
preview_dir.mkdir(parents=True, exist_ok=True)

# Thong ke
label_counts = Counter()
sample_counts = Counter()
invalid_images = []

# Duyet toan bo dataset
print("\nDang kiem tra anh va thong ke nhan...")

for index, item in enumerate(data):
    label = int(item["label"])
    label_counts[label] += 1

    try:
        image = item["image"].convert("RGB")

        # Luu toi da 3 anh mau cho moi nhan
        if sample_counts[label] < 3:
            image.save(
                preview_dir / f"label_{label}_sample_{sample_counts[label]}.jpg"
            )
            sample_counts[label] += 1

    except Exception as error:
        invalid_images.append((index, str(error)))

    if (index + 1) % 5000 == 0:
        print(f"Da kiem tra {index + 1}/{len(data)} anh")

# Luu thong ke CSV
csv_path = Path("dataset/class_distribution.csv")

with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["label_id", "image_count"])

    for label in sorted(label_counts):
        writer.writerow([label, label_counts[label]])

# In ket qua
print("\n========== CLASS DISTRIBUTION ==========")

for label in sorted(label_counts):
    print(f"Label {label:02d}: {label_counts[label]} images")

print("\nSo nhan:", len(label_counts))
print("Tong anh:", sum(label_counts.values()))
print("Anh loi:", len(invalid_images))
print("Anh mau:", sum(sample_counts.values()))

print("\nFile thong ke:", csv_path)
print("Thu muc anh mau:", preview_dir)

if invalid_images:
    print("\n========== INVALID IMAGES ==========")
    for item in invalid_images[:20]:
        print(item)

print("\nHOAN THANH KIEM TRA DATASET!")
