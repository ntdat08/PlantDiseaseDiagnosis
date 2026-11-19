import json
from pathlib import Path

files = [
    "knowledge_base/documents/diseases/apple_diseases.json",
    "knowledge_base/documents/diseases/tomato_diseases.json"
]

for file_path in files:
    print("\n" + "=" * 70)
    print("FILE:", file_path)
    print("=" * 70)

    try:
        data = json.loads(
            Path(file_path).read_text(encoding="utf-8")
        )

        print("JSON: VALID")
        print("Total records:", len(data))

        for item in data:
            print("\nDisease ID:", item.get("id"))
            print("Plant:", item.get("plant"))
            print("Disease:", item.get("disease"))
            print("Symptoms:", item.get("symptoms", "")[:300])
            print("Causes:", item.get("causes", "")[:200])
            print("Prevention:", item.get("prevention", "")[:150])
            print("Treatment:", item.get("treatment", "")[:150])

    except Exception as error:
        print("ERROR:", error)

print("\nJSON inspection completed!")
