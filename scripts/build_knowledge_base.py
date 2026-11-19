import json
import os

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JSON_PATH = os.path.join(
    BASE_DIR,
    "knowledge_base",
    "documents",
    "diseases",
    "tomato_diseases.json"
)

VECTOR_DB_PATH = os.path.join(
    BASE_DIR,
    "vector_store"
)

COLLECTION_NAME = "plant_disease_knowledge"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ==========================================
# 2. LOAD KNOWLEDGE DOCUMENTS
# ==========================================

def load_documents():

    print("Loading knowledge documents...")

    with open(JSON_PATH, "r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list) or not documents:
        raise ValueError("Knowledge document list is empty or invalid.")

    required_fields = [
        "id",
        "plant",
        "disease",
        "symptoms",
        "causes",
        "prevention",
        "treatment"
    ]

    for document in documents:
        for field in required_fields:
            if field not in document or not str(document[field]).strip():
                raise ValueError(
                    f"Document {document.get('id', 'unknown')} "
                    f"is missing field: {field}"
                )

    print(f"Loaded {len(documents)} documents.")

    return documents


# ==========================================
# 3. CONVERT DOCUMENTS INTO TEXT
# ==========================================

def create_text_documents(documents):

    texts = []
    ids = []
    metadatas = []

    for document in documents:

        text = (
            f"Plant: {document['plant']}\n"
            f"Disease: {document['disease']}\n"
            f"Symptoms: {document['symptoms']}\n"
            f"Causes: {document['causes']}\n"
            f"Prevention: {document['prevention']}\n"
            f"Treatment: {document['treatment']}"
        )

        texts.append(text)

        ids.append(document["id"])

        metadatas.append({
            "plant": document["plant"],
            "disease": document["disease"],
            "source_file": "tomato_diseases.json"
        })

    return texts, ids, metadatas


# ==========================================
# 4. CREATE EMBEDDINGS
# ==========================================

def create_embeddings(texts):

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Embedding shape:", embeddings.shape)

    return embeddings


# ==========================================
# 5. STORE DATA IN CHROMADB
# ==========================================

def save_to_chromadb(texts, ids, metadatas, embeddings):

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=VECTOR_DB_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Remove old records with matching IDs before updating
    existing = collection.get(ids=ids)

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    print("\nKnowledge stored successfully!")
    print("Collection:", COLLECTION_NAME)
    print("Total documents:", collection.count())


# ==========================================
# 6. MAIN
# ==========================================

def main():

    documents = load_documents()

    texts, ids, metadatas = create_text_documents(documents)

    embeddings = create_embeddings(texts)

    save_to_chromadb(
        texts,
        ids,
        metadatas,
        embeddings
    )

    print("\nKnowledge base construction completed!")


if __name__ == "__main__":
    main()
