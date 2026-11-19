import os
import chromadb


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store")

COLLECTION_NAME = "plant_disease_knowledge_chunks"


def main():

    client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print("=" * 60)
    print("KNOWLEDGE BASE INSPECTION")
    print("=" * 60)

    print("\nTotal chunks:", collection.count())

    results = collection.get(
        include=["documents", "metadatas"]
    )

    # Count chunks by plant
    plant_counts = {}

    for metadata in results["metadatas"]:

        plant = metadata["plant"]

        plant_counts[plant] = plant_counts.get(plant, 0) + 1

    print("\nChunks by plant:")

    for plant, count in sorted(plant_counts.items()):
        print(f"- {plant}: {count}")

    # Count chunks by source file
    source_counts = {}

    for metadata in results["metadatas"]:

        source_file = metadata["source_file"]

        source_counts[source_file] = (
            source_counts.get(source_file, 0) + 1
        )

    print("\nChunks by source file:")

    for filename, count in sorted(source_counts.items()):
        print(f"- {filename}: {count}")

    # Show sample chunks
    print("\nSample chunks:")

    for i in range(min(5, len(results["ids"]))):

        print("\nID:", results["ids"][i])
        print("Metadata:", results["metadatas"][i])
        print("Document:", results["documents"][i][:250])

    print("\nInspection completed!")


if __name__ == "__main__":
    main()
