from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

sentences = [
    "Lá cà chua xuất hiện những đốm nâu và vàng.",
    "Bệnh đốm lá sớm trên cây cà chua.",
    "Cây lúa cần được tưới nước thường xuyên."
]

print("\nGenerating embeddings...")

embeddings = model.encode(
    sentences,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("\n========== RESULTS ==========")
print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)
print("Data type:", embeddings.dtype)
print("Vector dimension:", embeddings.shape[1])

similarity = np.dot(
    embeddings[0],
    embeddings[1]
) / (
    np.linalg.norm(embeddings[0]) *
    np.linalg.norm(embeddings[1])
)

print("\nSimilarity between sentence 1 and 2:", round(float(similarity), 4))

print("\nEmbedding test completed successfully!")