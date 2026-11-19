import os
import chromadb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_store")

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

collection = client.get_collection(
    name="plant_disease_knowledge"
)

print("========== CHROMADB INFORMATION ==========")
print("Collection:", collection.name)
print("Total documents:", collection.count())

print("\n========== STORED DOCUMENTS ==========")

results = collection.get(
    include=["documents", "metadatas"]
)

for i, document_id in enumerate(results["ids"]):
    print("\nID:", document_id)
    print("Metadata:", results["metadatas"][i])
    print("Content:", results["documents"][i][:200], "...")


print("\nChromaDB test completed!")
