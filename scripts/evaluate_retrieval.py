import os
import json
import csv

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store")

TEST_CASES_PATH = os.path.join(
    BASE_DIR,
    "evaluation",
    "rag",
    "retrieval_test_cases.json"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "evaluation",
    "rag",
    "retrieval_results.csv"
)

COLLECTION_NAME = "plant_disease_knowledge"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 3


# ==========================================
# 2. LOAD COMPONENTS
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

client = chromadb.PersistentClient(
    path=VECTOR_DB_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Collection:", collection.name)
print("Knowledge documents:", collection.count())


# ==========================================
# 3. LOAD TEST CASES
# ==========================================

with open(TEST_CASES_PATH, "r", encoding="utf-8") as file:
    test_cases = json.load(file)

print("Test cases:", len(test_cases))


# ==========================================
# 4. EVALUATE RETRIEVAL
# ==========================================

results_list = []

correct_at_1 = 0
correct_at_3 = 0

for index, test_case in enumerate(test_cases):

    question = test_case["question"]
    expected = test_case["expected_disease"]

    query_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(TOP_K, collection.count()),
        include=["metadatas", "distances"]
    )

    retrieved_diseases = [
        metadata["disease"]
        for metadata in results["metadatas"][0]
    ]

    distances = results["distances"][0]

    rank_1 = retrieved_diseases[0]

    hit_at_1 = int(rank_1 == expected)
    hit_at_3 = int(expected in retrieved_diseases)

    correct_at_1 += hit_at_1
    correct_at_3 += hit_at_3

    results_list.append({
        "question": question,
        "expected_disease": expected,
        "top_1_disease": rank_1,
        "top_1_distance": round(distances[0], 4),
        "retrieved_diseases": " | ".join(retrieved_diseases),
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3
    })

    print(f"\nQuestion {index + 1}: {question}")
    print("Expected:", expected)
    print("Retrieved:", retrieved_diseases)
    print("Hit@1:", hit_at_1)
    print("Hit@3:", hit_at_3)


# ==========================================
# 5. CALCULATE METRICS
# ==========================================

total = len(test_cases)

hit_rate_1 = correct_at_1 / total
hit_rate_3 = correct_at_3 / total

print("\n" + "=" * 55)
print("RETRIEVAL EVALUATION RESULTS")
print("=" * 55)

print("Total questions:", total)
print(f"Hit@1: {hit_rate_1:.4f} ({hit_rate_1 * 100:.2f}%)")
print(f"Hit@3: {hit_rate_3:.4f} ({hit_rate_3 * 100:.2f}%)")


# ==========================================
# 6. SAVE RESULTS
# ==========================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results_list[0].keys()
    )

    writer.writeheader()
    writer.writerows(results_list)

print("\nResults saved to:", OUTPUT_PATH)
print("Evaluation completed successfully!")
