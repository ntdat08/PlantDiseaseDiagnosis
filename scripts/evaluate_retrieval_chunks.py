import json
import csv
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "evaluation" / "rag" / "retrieval_test_cases.json"
CHROMA_DIR = BASE_DIR / "vector_store"

COLLECTION_NAME = "plant_disease_knowledge_chunks"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_CHUNKS = 12

OUTPUT_FILE = (
    BASE_DIR / "evaluation" / "rag" / "retrieval_chunks_results.csv"
)


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def normalize_text(text):

    text = str(text).lower().strip()
    text = re.sub(r"[_\s]+", " ", text)

    return text


def get_test_fields(item):

    question = item.get("question")
    expected_plant = item.get("expected_plant")
    expected_disease = item.get("expected_disease")

    if not question or not expected_plant or not expected_disease:

        raise ValueError(
            "Each test case must contain question, "
            "expected_plant and expected_disease."
        )

    return (
        str(question),
        str(expected_plant),
        str(expected_disease)
    )


def is_match(actual, expected):

    actual = normalize_text(actual)
    expected = normalize_text(expected)

    return (
        actual == expected
        or expected in actual
        or actual in expected
    )


# ==========================================
# 3. LOAD TEST CASES
# ==========================================

print("Loading test cases...")

with open(TEST_FILE, "r", encoding="utf-8") as file:

    test_cases = json.load(file)


print("Total questions:", len(test_cases))


if not test_cases:

    raise ValueError("No test cases found!")


# ==========================================
# 4. LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)


# ==========================================
# 5. CONNECT CHROMADB
# ==========================================

print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

total_chunks = collection.count()

print("Collection:", COLLECTION_NAME)
print("Total chunks:", total_chunks)


if total_chunks == 0:

    raise ValueError("Knowledge collection is empty!")


# ==========================================
# 6. EVALUATE RETRIEVAL
# ==========================================

results = []

hit_at_1 = 0
hit_at_3 = 0

plant_hit_at_1 = 0
plant_hit_at_3 = 0

print("\nEvaluating plant and disease retrieval...\n")


for index, item in enumerate(test_cases, start=1):

    question, expected_plant, expected_disease = (
        get_test_fields(item)
    )

    # Generate query embedding
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    # Retrieve top chunks
    retrieved = collection.query(
        query_embeddings=query_embedding,
        n_results=min(TOP_CHUNKS, total_chunks),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = retrieved["documents"][0]
    metadatas = retrieved["metadatas"][0]
    distances = retrieved["distances"][0]

    # Aggregate chunks by plant + disease
    disease_scores = {}

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        plant = metadata.get("plant", "Unknown")
        disease = metadata.get("disease", "Unknown")

        key = (plant, disease)

        if (
            key not in disease_scores
            or distance < disease_scores[key]["distance"]
        ):

            disease_scores[key] = {
                "distance": distance,
                "section": metadata.get("section", "Unknown"),
                "document": document
            }

    # Rank plant-disease pairs
    ranked_diseases = sorted(
        disease_scores.items(),
        key=lambda item: item[1]["distance"]
    )

    ranked_pairs = [
        (plant, disease)
        for (plant, disease), score in ranked_diseases
    ]

    expected_plant_normalized = normalize_text(expected_plant)
    expected_disease_normalized = normalize_text(expected_disease)

    # Match requires BOTH plant and disease to be correct
    def pair_match(pair):

        actual_plant, actual_disease = pair

        return (
            normalize_text(actual_plant) == expected_plant_normalized
            and normalize_text(actual_disease) == expected_disease_normalized
        )

    top1_match = (
        len(ranked_pairs) > 0
        and pair_match(ranked_pairs[0])
    )

    top3_match = any(
        pair_match(pair)
        for pair in ranked_pairs[:3]
    )

    # Plant-only metrics
    plant_top1_match = (
        len(ranked_pairs) > 0
        and normalize_text(ranked_pairs[0][0])
        == expected_plant_normalized
    )

    plant_top3_match = any(
        normalize_text(pair[0]) == expected_plant_normalized
        for pair in ranked_pairs[:3]
    )

    if top1_match:
        hit_at_1 += 1

    if top3_match:
        hit_at_3 += 1

    if plant_top1_match:
        plant_hit_at_1 += 1

    if plant_top3_match:
        plant_hit_at_3 += 1

    # Top 1 result
    if ranked_pairs:

        top1_plant, top1_disease = ranked_pairs[0]

        top1_section = disease_scores[
            (top1_plant, top1_disease)
        ]["section"]

    else:

        top1_plant = "None"
        top1_disease = "None"
        top1_section = "None"

    top3_display = " | ".join(
        [
            f"{plant}: {disease}"
            for plant, disease in ranked_pairs[:3]
        ]
    )

    print(f"Question {index}: {question}")

    print(
        f"Expected: {expected_plant} - {expected_disease}"
    )

    print(
        f"Top 1: {top1_plant} - {top1_disease} "
        f"({top1_section})"
    )

    print(f"Top 3: {top3_display}")

    print(
        f"Hit@1: {int(top1_match)} | "
        f"Hit@3: {int(top3_match)}"
    )

    print("-" * 75)

    results.append({

        "question": question,

        "expected_plant": expected_plant,

        "expected_disease": expected_disease,

        "top1_plant": top1_plant,

        "top1_disease": top1_disease,

        "top1_section": top1_section,

        "top3_results": top3_display,

        "hit_at_1": int(top1_match),

        "hit_at_3": int(top3_match),

        "plant_hit_at_1": int(plant_top1_match),

        "plant_hit_at_3": int(plant_top3_match)

    })


# ==========================================
# 7. CALCULATE METRICS
# ==========================================

total_questions = len(test_cases)

hit1_rate = hit_at_1 / total_questions
hit3_rate = hit_at_3 / total_questions

plant_hit1_rate = plant_hit_at_1 / total_questions
plant_hit3_rate = plant_hit_at_3 / total_questions


print("\n========== RETRIEVAL EVALUATION ==========")

print("Total questions:", total_questions)

print("\nPlant + Disease:")
print(
    f"Hit@1: {hit1_rate:.4f} "
    f"({hit1_rate * 100:.2f}%)"
)

print(
    f"Hit@3: {hit3_rate:.4f} "
    f"({hit3_rate * 100:.2f}%)"
)

print("\nPlant only:")
print(
    f"Plant Hit@1: {plant_hit1_rate:.4f} "
    f"({plant_hit1_rate * 100:.2f}%)"
)

print(
    f"Plant Hit@3: {plant_hit3_rate:.4f} "
    f"({plant_hit3_rate * 100:.2f}%)"
)

print("==========================================")


# ==========================================
# 8. SAVE RESULTS
# ==========================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "question",
            "expected_plant",
            "expected_disease",
            "top1_plant",
            "top1_disease",
            "top1_section",
            "top3_results",
            "hit_at_1",
            "hit_at_3",
            "plant_hit_at_1",
            "plant_hit_at_3"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print("\nResults saved to:", OUTPUT_FILE)

print("Retrieval evaluation completed!")
