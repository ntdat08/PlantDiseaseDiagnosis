
import json

files = [
    "knowledge_base/documents/diseases/tomato_diseases.json",
    "knowledge_base/documents/diseases/apple_diseases.json"
]

for file_path in files:
    print("\n" + "=" * 70)
    print("FILE:", file_path)
    print("=" * 70)

    with open(file_path, "r", encoding="utf-8") as f:
        diseases = json.load(f)

    for disease in diseases:
        if disease["disease"] == "healthy":
            continue

        print("\nDisease:", disease["disease"])
        print("Symptoms:")
        print(disease["symptoms"])
        print("-" * 70)
