from datasets import load_dataset
from collections import Counter
from pathlib import Path

print("Dang tai PlantVillage tu DPDL Benchmark...")

dataset = load_dataset(
    "dpdl-benchmark/plant_village",
    cache_dir="dataset/hf_cache"
)

print("\n========== DATASET INFORMATION ==========")
print(dataset)

# Lay tap train
train_data = dataset["train"]

# Kiem tra cau truc
print("\n========== FEATURES ==========")
print(train_data.features)

print("\nSo luong anh:", len(train_data))
print("Cac cot:", train_data.column_names)

# Kiem tra mot mau
sample = train_data[0]

print("\n========== SAMPLE ==========")
print("Cac truong:", sample.keys())
print("Kieu anh:", type(sample["image"]))
print("Kich thuoc anh:", sample["image"].size)
print("Nhan:", sample["label"])

# Thong ke nhan
label_counts = Counter(train_data["label"])

print("\n========== LABEL DISTRIBUTION ==========")

label_feature = train_data.features["label"]

if hasattr(label_feature, "names"):
    label_names = label_feature.names

    print("So lop:", len(label_names))

    for index, name in enumerate(label_names):
        print(f"{index}: {name} - {label_counts[index]} anh")
else:
    print("Nhan chua co ten lop trong ClassLabel.")
    print("Cac nhan:", sorted(label_counts.keys()))

# Luu mot so anh mau
preview_dir = Path("dataset/preview")
preview_dir.mkdir(parents=True, exist_ok=True)

for i in range(min(10, len(train_data))):
    image = train_data[i]["image"].convert("RGB")
    image.save(preview_dir / f"sample_{i}.jpg")

print("\nDa luu 10 anh mau tai:", preview_dir)
print("HOAN THANH KIEM TRA!")
