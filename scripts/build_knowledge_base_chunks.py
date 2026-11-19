from pathlib import Path
from datetime import datetime
import json
import shutil

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DISEASE_DIR = BASE_DIR / "knowledge_base" / "documents" / "diseases"
VECTOR_DB_PATH = BASE_DIR / "vector_store"

COLLECTION_NAME = "plant_disease_knowledge_chunks"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

REQUIRED_FIELDS = [
    "symptoms",
    "causes",
    "prevention",
    "treatment"
]

EXPECTED_RECORDS = 38
EXPECTED_CHUNKS = EXPECTED_RECORDS * len(REQUIRED_FIELDS)


# ==========================================
# 2. LOAD AND VALIDATE DOCUMENTS
# ==========================================

def load_documents():

    print("Loading disease documents...")

    json_files = sorted(DISEASE_DIR.glob("*_diseases.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No disease JSON files found in: {DISEASE_DIR}"
        )

    all_documents = []

    for json_path in json_files:

        print(f"\nReading: {json_path.name}")

        with open(json_path, "r", encoding="utf-8-sig") as file:
            documents = json.load(file)

        if not isinstance(documents, list):
            raise ValueError(f"Invalid JSON structure: {json_path.name}")

        for document in documents:
            document["_source_file"] = json_path.name
            all_documents.append(document)

        print("Records:", len(documents))

    print("\nTotal disease records:", len(all_documents))

    if len(all_documents) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} records, "
            f"found {len(all_documents)}"
        )

    ids = [document.get("id") for document in all_documents]

    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate disease IDs.")

    for document in all_documents:

        for field in ["id", "plant", "disease", "source"] + REQUIRED_FIELDS:

            value = document.get(field)

            if value is None or not str(value).strip():
                raise ValueError(
                    f"Missing field '{field}' in {document.get('id')}"
                )

    print("Document validation: PASSED")

    return all_documents


# ==========================================
# 3. CREATE CHUNKS
# ==========================================

def create_chunks(documents):

    chunks = []
    ids = []
    metadatas = []

    for document in documents:

        for field in REQUIRED_FIELDS:

            content = str(document[field]).strip()

            chunk_id = f"{document['id']}_{field}"

            chunks.append(content)
            ids.append(chunk_id)

            metadatas.append({
                "plant": document["plant"],
                "disease": document["disease"],
                "section": field,
                "source_file": document["_source_file"],
                "source": str(document["source"])
            })

    if len(chunks) != EXPECTED_CHUNKS:
        raise ValueError(
            f"Expected {EXPECTED_CHUNKS} chunks, found {len(chunks)}"
        )

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    print("\nTotal chunks created:", len(chunks))
    print("Chunk validation: PASSED")

    return chunks, ids, metadatas


# ==========================================
# 4. GENERATE EMBEDDINGS
# ==========================================

def generate_embeddings(chunks):

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("Generating embeddings...")

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )

    print("Embedding shape:", embeddings.shape)

    if embeddings.shape[0] != EXPECTED_CHUNKS:
        raise ValueError("Incorrect number of embeddings.")

    if embeddings.shape[1] != 384:
        raise ValueError(
            f"Expected embedding dimension 384, got {embeddings.shape[1]}"
        )

    return embeddings


# ==========================================
# 5. REBUILD CHROMADB
# ==========================================

def save_to_chromadb(chunks, ids, metadatas, embeddings):

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH)
    )

    # Xóa collection cũ để không giữ chunks lỗi thời
    existing = [
        item.name for item in client.list_collections()
    ]

    if COLLECTION_NAME in existing:
        print("Deleting old collection...")
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("Saving new collection...")

    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    actual_count = collection.count()

    print("\n========== CHROMADB RESULT ==========")
    print("Collection:", collection.name)
    print("Expected chunks:", EXPECTED_CHUNKS)
    print("Actual chunks:", actual_count)

    if actual_count != EXPECTED_CHUNKS:
        raise RuntimeError("ChromaDB chunk count mismatch.")

    print("CHROMADB REBUILD: SUCCESS")


# ==========================================
# 6. MAIN
# ==========================================

def main():

    start_time = datetime.now()

    documents = load_documents()

    chunks, ids, metadatas = create_chunks(documents)

    embeddings = generate_embeddings(chunks)

    save_to_chromadb(
        chunks,
        ids,
        metadatas,
        embeddings
    )

    elapsed = datetime.now() - start_time

    print("\n========== BUILD COMPLETED ==========")
    print("Total records:", len(documents))
    print("Total chunks:", len(chunks))
    print("Elapsed time:", elapsed)


if __name__ == "__main__":
    main()
