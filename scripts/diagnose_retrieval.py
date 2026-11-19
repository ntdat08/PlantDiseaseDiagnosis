import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "evaluation" / "rag" / "retrieval_test_cases.json"
CHROMA_DIR = BASE_DIR / "vector_store"

COLLECTION_NAME = "plant_disease_knowledge_chunks"

MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 8

# Các câu đang có Hit@1 = 0
FAILED_QUESTIONS = [2, 3, 11, 13, 17]

def main():

    print("Loading test cases...")

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print("\n" + "=" * 80)
    print("RETRIEVAL DIAGNOSTICS")
    print("=" * 80)

    for question_number in FAILED_QUESTIONS:

        item = test_cases[question_number - 1]

        question = item["question"]

        expected_plant = item["expected_plant"]
        expected_disease = item["expected_disease"]

        query_embedding = model.encode(
            [question],
            normalize_embeddings=True
        ).tolist()

        retrieved = collection.query(
            query_embeddings=query_embedding,
            n_results=min(TOP_K, collection.count()),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        print(f"\nQUESTION {question_number}")
        print("Question:", question)
        print("Expected:", expected_plant, "-", expected_disease)

        print("\nRetrieved chunks:")

        for rank, (document, metadata, distance) in enumerate(
            zip(
                retrieved["documents"][0],
                retrieved["metadatas"][0],
                retrieved["distances"][0]
            ),
            start=1
        ):

            plant = metadata.get("plant")
            disease = metadata.get("disease")
            section = metadata.get("section")

            is_correct = (
                plant == expected_plant
                and disease == expected_disease
            )

            status = "CORRECT" if is_correct else ""

            print(f"\nRank {rank} | Distance: {distance:.4f} {status}")
            print(f"Plant: {plant}")
            print(f"Disease: {disease}")
            print(f"Section: {section}")
            print("Content:", document[:400])

        print("\n" + "-" * 80)

    print("\nDiagnostics completed!")


if __name__ == "__main__":
    main()
