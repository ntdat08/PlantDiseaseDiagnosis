from datasets import load_dataset
from collections import defaultdict

print("Dang doc PlantVillage...")

dataset = load_dataset(
    "dpdl-benchmark/plant_village",
    cache_dir="dataset/hf_cache"
)

data = dataset["train"]

# Luu ten file theo tung label
label_files = defaultdict(list)

for sample in data:
    label = sample["label"]
    filename = sample["image/filename"]

    if len(label_files[label]) < 5:
        label_files[label].append(filename)

print("\n========== KIEM TRA TEN LOP ==========")

for label in sorted(label_files.keys()):
    print(f"\nLABEL {label:02d}")

    for filename in label_files[label]:
        print("  ", filename)

print("\nHOAN THANH!")
