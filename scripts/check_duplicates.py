import hashlib
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
from datasets import load_dataset

from class_names import CLASS_NAMES


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_NAME = "dpdl-benchmark/plant_village"
CACHE_DIR = "dataset/hf_cache"

OUTPUT_DIR = Path("evaluation/classification")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Dang tai metadata PlantVillage...")

dataset = load_dataset(
    DATASET_NAME,
    cache_dir=CACHE_DIR
)

data = dataset["train"]

total_images = len(data)

print(f"Tong so anh: {total_images}")
print("Bat dau kiem tra anh trung lap...\n")


# ==========================================
# 2. HASH IMAGE CONTENT
# ==========================================

hash_groups = defaultdict(list)

start_time = time.time()

for index, item in enumerate(data):

    image = item["image"]
    label_id = int(item["label"])

    # Chuyen anh ve RGB de dam bao cung dinh dang
    image = image.convert("RGB")

    # Tinh SHA256 dua tren noi dung pixel
    image_hash = hashlib.sha256(
        image.tobytes()
    ).hexdigest()

    filename = item["image/filename"]

    if isinstance(filename, bytes):
        filename = filename.decode(
            "utf-8",
            errors="replace"
        )

    hash_groups[image_hash].append({
        "dataset_index": index,
        "label_id": label_id,
        "class_name": CLASS_NAMES[label_id],
        "filename": filename
    })

    # Hien thi tien do
    if (index + 1) % 5000 == 0:
        elapsed = time.time() - start_time

        print(
            f"Da kiem tra {index + 1}/{total_images} anh "
            f"({elapsed:.1f} giay)"
        )


# ==========================================
# 3. FIND DUPLICATE GROUPS
# ==========================================

duplicate_groups = []
duplicate_records = []
all_records = []

for image_hash, records in hash_groups.items():

    # Luu thong tin hash cua tat ca anh
    for record in records:
        all_records.append({
            **record,
            "image_hash": image_hash
        })

    # Chi lay nhom co tu 2 anh tro len
    if len(records) > 1:

        labels = sorted(set(
            record["label_id"]
            for record in records
        ))

        duplicate_groups.append({
            "image_hash": image_hash,
            "duplicate_count": len(records),
            "different_labels": len(labels) > 1,
            "label_ids": ", ".join(map(str, labels))
        })

        for record in records:
            duplicate_records.append({
                **record,
                "image_hash": image_hash,
                "group_size": len(records)
            })


# ==========================================
# 4. STATISTICS
# ==========================================

duplicate_image_count = sum(
    group["duplicate_count"]
    for group in duplicate_groups
)

duplicate_excess = sum(
    group["duplicate_count"] - 1
    for group in duplicate_groups
)

cross_label_groups = sum(
    1 for group in duplicate_groups
    if group["different_labels"]
)

unique_images = len(hash_groups)

elapsed = time.time() - start_time

print("\n========== DUPLICATE CHECK SUMMARY ==========")

print(f"Total images: {total_images}")
print(f"Unique image contents: {unique_images}")
print(f"Duplicate groups: {len(duplicate_groups)}")
print(f"Images belonging to duplicate groups: {duplicate_image_count}")
print(f"Excess duplicate images: {duplicate_excess}")
print(f"Duplicate groups across different labels: {cross_label_groups}")
print(f"Processing time: {elapsed / 60:.2f} minutes")


# ==========================================
# 5. SAVE RESULTS
# ==========================================

hash_df = pd.DataFrame(all_records)

duplicate_df = pd.DataFrame(duplicate_records)

group_df = pd.DataFrame(duplicate_groups)

hash_path = OUTPUT_DIR / "image_hashes.csv"
duplicate_path = OUTPUT_DIR / "duplicate_images.csv"
group_path = OUTPUT_DIR / "duplicate_groups.csv"

hash_df.to_csv(hash_path, index=False, encoding="utf-8-sig")
duplicate_df.to_csv(duplicate_path, index=False, encoding="utf-8-sig")
group_df.to_csv(group_path, index=False, encoding="utf-8-sig")

print("\n========== SAVED FILES ==========")
print(f"All image hashes: {hash_path}")
print(f"Duplicate images: {duplicate_path}")
print(f"Duplicate groups: {group_path}")

print("\nHOAN THANH KIEM TRA!")
