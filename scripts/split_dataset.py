from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ==========================================
# 1. CONFIGURATION
# ==========================================

INPUT_FILE = Path(
    "evaluation/classification/image_hashes.csv"
)

OUTPUT_DIR = Path("dataset/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("Bat dau chia du lieu PlantVillage...")


# ==========================================
# 2. LOAD IMAGE HASHES
# ==========================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Khong tim thay file: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "dataset_index",
    "label_id",
    "class_name",
    "filename",
    "image_hash"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Thieu cot du lieu: {column}"
        )

df["label_id"] = df["label_id"].astype(int)
df["dataset_index"] = df["dataset_index"].astype(int)

print(f"Tong so anh ban dau: {len(df)}")


# ==========================================
# 3. CHECK DUPLICATE LABELS
# ==========================================

labels_per_hash = df.groupby(
    "image_hash"
)["label_id"].nunique()

cross_label_duplicates = labels_per_hash[
    labels_per_hash > 1
]

if len(cross_label_duplicates) > 0:
    raise ValueError(
        "Phat hien anh trung lap nhung khac nhan. "
        "Can kiem tra truoc khi chia du lieu."
    )


# ==========================================
# 4. REMOVE EXACT DUPLICATES
# ==========================================

df_unique = df.drop_duplicates(
    subset="image_hash",
    keep="first"
).copy()

df_unique = df_unique.reset_index(drop=True)

print(f"So anh sau khi loai trung: {len(df_unique)}")
print(f"So anh duoc loai: {len(df) - len(df_unique)}")
print(f"So lop: {df_unique['label_id'].nunique()}")


# ==========================================
# 5. SPLIT TRAIN / VALIDATION / TEST
# ==========================================

train_df, temp_df = train_test_split(
    df_unique,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df_unique["label_id"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label_id"]
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"


# ==========================================
# 6. COMBINE DATA
# ==========================================

split_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

split_df = split_df.sort_values(
    by="dataset_index"
).reset_index(drop=True)


# ==========================================
# 7. VERIFY DATA SPLIT
# ==========================================

assert len(split_df) == len(df_unique)

assert split_df["image_hash"].is_unique

assert set(train_df["image_hash"]).isdisjoint(
    set(val_df["image_hash"])
)

assert set(train_df["image_hash"]).isdisjoint(
    set(test_df["image_hash"])
)

assert set(val_df["image_hash"]).isdisjoint(
    set(test_df["image_hash"])
)

expected_labels = set(df_unique["label_id"])

for split_name, split_data in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    actual_labels = set(split_data["label_id"])

    if actual_labels != expected_labels:
        raise ValueError(
            f"{split_name} khong co du 38 lop!"
        )


# ==========================================
# 8. SAVE SPLIT MANIFEST
# ==========================================

manifest_path = OUTPUT_DIR / "split_manifest.csv"

split_df.to_csv(
    manifest_path,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 9. CLASS DISTRIBUTION BY SPLIT
# ==========================================

summary = split_df.pivot_table(
    index=["label_id", "class_name"],
    columns="split",
    values="dataset_index",
    aggfunc="count",
    fill_value=0
).reset_index()

for column in ["train", "validation", "test"]:
    if column not in summary.columns:
        summary[column] = 0

summary["total"] = (
    summary["train"]
    + summary["validation"]
    + summary["test"]
)

summary_path = OUTPUT_DIR / "split_class_distribution.csv"

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 10. PRINT RESULTS
# ==========================================

print("\n========== DATA SPLIT SUMMARY ==========")

total = len(split_df)

for name, data in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:
    count = len(data)
    percentage = count / total * 100

    print(
        f"{name}: {count} images "
        f"({percentage:.2f}%)"
    )

print(f"\nTotal images: {total}")
print(f"Total classes: {split_df['label_id'].nunique()}")

print("\n========== CLASS DISTRIBUTION ==========")
print(summary.to_string(index=False))

print("\n========== SAVED FILES ==========")
print(f"Split manifest: {manifest_path}")
print(f"Class distribution: {summary_path}")

print("\n========== VERIFICATION ==========")
print("No duplicate image hashes across splits.")
print("All 38 classes exist in all three splits.")
print("Data split verification: PASSED")

print("\nHOAN THANH CHIA DU LIEU!")
