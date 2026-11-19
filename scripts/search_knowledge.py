import os
import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store")

COLLECTION_NAME = "plant_disease_knowledge"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TOP_K = 3


# ==========================================
# 2. LOAD MODELS AND VECTOR DATABASE
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=VECTOR_DB_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Collection loaded:", collection.name)
print("Total documents:", collection.count())


# ==========================================
# 3. SEARCH KNOWLEDGE
# ==========================================

def search_knowledge(question, top_k=TOP_K):

    print("\nSearching for relevant knowledge...")

    # Convert question into embedding vector
    query_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Search similar documents
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results


# ==========================================
# 4. DISPLAY RESULTS
# ==========================================

def display_results(question, results):

    print("\n" + "=" * 60)
    print("USER QUESTION:")
    print(question)

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS:")

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):

        print("\nRank:", i + 1)
        print("Document ID:", ids[i])
        print("Plant:", metadatas[i]["plant"])
        print("Disease:", metadatas[i]["disease"])
        print("Distance:", round(distances[i], 4))

        print("\nRetrieved content:")
        print(documents[i])

        print("-" * 60)


# ==========================================
# 5. MAIN
# ==========================================

def main():

    question = input(
        "\nEnter your plant disease question: "
    ).strip()

    if not question:
        print("Question cannot be empty.")
        return

    results = search_knowledge(question)

    display_results(question, results)


if __name__ == "__main__":
    main()
