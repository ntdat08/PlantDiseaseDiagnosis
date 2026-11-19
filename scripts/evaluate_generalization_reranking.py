import json
import csv
import re
import unicodedata
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = (
    BASE_DIR / "evaluation" / "rag" / "retrieval_generalization_test_cases.json"
)

CHROMA_DIR = BASE_DIR / "vector_store"

COLLECTION_NAME = "plant_disease_knowledge_chunks"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_CHUNKS = 12

OUTPUT_FILE = (
    BASE_DIR / "evaluation" / "rag"
    / "generalization_reranking_results.csv"
)


# ==========================================
# 2. DISEASE-SPECIFIC KEYWORDS
# ==========================================

# CÃ¡c dáº¥u hiá»‡u Ä‘áº·c trÆ°ng dÃ¹ng Ä‘á»ƒ thá»­ nghiá»‡m.
# Chá»‰ cá»™ng Ä‘iá»ƒm khi tá»« khÃ³a xuáº¥t hiá»‡n trong cÃ¢u há»i.

DISEASE_RULES = {

    ("Tomato", "Early blight"): {
        "strong": [
            "vòng tròn đồng tâm",
            "vòng đồng tâm",
            "hình bia bắn",
            "đốm có vòng đồng tâm",
            "concentric rings",
            "target pattern",
            "alternaria solani"
        ],
        "medium": [
            "lá già",
            "đốm vòng",
            "quầng vàng",
            "đốm nâu sẫm",
            "older leaves",
            "yellow halos"
        ]
    },

    ("Tomato", "Late blight"): {
        "strong": [
            "mốc trắng",
            "lớp mốc trắng",
            "lan rộng nhanh",
            "vết bệnh lớn",
            "phytophthora infestans",
            "rapidly expanding lesions",
            "white mold-like growth",
            "water-soaked lesions"
        ],
        "medium": [
            "mốc sương",
            "mát và ẩm",
            "nâu đen",
            "úng nước",
            "cool and humid",
            "dark brown lesions",
            "black lesions"
        ]
    },

    ("Tomato", "Bacterial spot"): {
        "strong": [
            "nhiều đốm nhỏ",
            "đốm nhỏ riêng biệt",
            "đốm nhỏ rải rác",
            "đốm li ti",
            "xanthomonas",
            "numerous small spots",
            "small scattered spots",
            "separate small lesions"
        ],
        "medium": [
            "đốm vi khuẩn",
            "chuyển sang màu nâu và khô",
            "đốm sẫm",
            "phân bố rải rác",
            "yellow halos",
            "turn brown and dry",
            "dark water-soaked spots"
        ]
    },

    ("Apple", "Apple scab"): {
        "strong": [
            "đóng vảy",
            "vết sần sùi",
            "vảy trên quả",
            "quả bị nứt",
            "venturia inaequalis",
            "corky scab lesions",
            "rough scab lesions",
            "olive-green spots"
        ],
        "medium": [
            "ghẻ táo",
            "xanh ô liu",
            "mép không đều",
            "quả biến dạng",
            "đốm nâu sẫm",
            "irregular edges",
            "deformed fruit",
            "dark corky lesions"
        ]
    },

    ("Apple", "Black rot"): {
        "strong": [
            "mắt ếch",
            "hình mắt ếch",
            "viền tím",
            "tâm màu nâu vàng",
            "botryosphaeria obtusa",
            "diplodia seriata",
            "frog-eye leaf spots",
            "purple border"
        ],
        "medium": [
            "thối đen",
            "vòng tròn màu nâu đồng tâm",
            "quả khô quắt",
            "quả teo lại",
            "quả bị thối",
            "concentric rings",
            "shriveled fruit",
            "mummified fruit"
        ]
    },

    ("Apple", "Cedar-apple rust"): {
        "strong": [
            "gỉ sắt tuyết tùng",
            "rỉ sắt tuyết tùng",
            "gymnosporangium",
            "ống màu cam",
            "cấu trúc dạng ống",
            "orange tubular structures",
            "fringed orange structures"
        ],
        "medium": [
            "đốm vàng cam",
            "đốm màu cam",
            "tua màu cam",
            "mặt dưới lá",
            "yellow-orange spots",
            "orange spots",
            "underside of leaves"
        ]
    }
}


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def normalize_text(text):

    text = str(text).lower().strip()

    # Normalize Vietnamese characters
    text = text.replace("đ", "d")

    text = unicodedata.normalize("NFD", text)

    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def get_test_fields(item):

    question = item.get("question")
    expected_plant = item.get("expected_plant")
    expected_disease = item.get("expected_disease")

    if not question or not expected_plant or not expected_disease:

        raise ValueError(
            "Missing question, expected_plant or expected_disease."
        )

    return (
        str(question),
        str(expected_plant),
        str(expected_disease)
    )


def rule_score(question, plant, disease):

    question = normalize_text(question)

    rules = DISEASE_RULES.get((plant, disease), {})

    score = 0
    matched_keywords = []

    # Strong keywords: 0.15 each
    for keyword in rules.get("strong", []):

        if normalize_text(keyword) in question:

            score += 0.15
            matched_keywords.append(keyword)

    # Medium keywords: 0.05 each
    for keyword in rules.get("medium", []):

        if normalize_text(keyword) in question:

            score += 0.05
            matched_keywords.append(keyword)

    # Avoid excessive keyword boosting
    score = min(score, 0.30)

    return score, matched_keywords


# ==========================================
# 4. LOAD TEST CASES
# ==========================================

print("Loading test cases...")

with open(TEST_FILE, "r", encoding="utf-8") as file:

    test_cases = json.load(file)

print("Total questions:", len(test_cases))


if not test_cases:

    raise ValueError("No test cases found!")


# ==========================================
# 5. LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)


# ==========================================
# 6. CONNECT CHROMADB
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
# 7. EVALUATE RERANKING
# ==========================================

results = []

hit_at_1 = 0
hit_at_3 = 0

plant_hit_at_1 = 0
plant_hit_at_3 = 0

print("\nEvaluating Embedding + Reranking...\n")


for index, item in enumerate(test_cases, start=1):

    question, expected_plant, expected_disease = (
        get_test_fields(item)
    )

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

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
                "distance": float(distance),
                "section": metadata.get("section", "Unknown"),
                "document": document
            }

    # --------------------------------------
    # Reranking
    # --------------------------------------

    ranked_diseases = []

    for (plant, disease), data in disease_scores.items():

        embedding_distance = data["distance"]

        keyword_bonus, matched_keywords = rule_score(
            question,
            plant,
            disease
        )

        # Lower score is better.
        # Distance is the original embedding distance.
        rerank_score = embedding_distance - keyword_bonus

        ranked_diseases.append({

            "plant": plant,

            "disease": disease,

            "distance": embedding_distance,

            "keyword_bonus": keyword_bonus,

            "rerank_score": rerank_score,

            "section": data["section"],

            "matched_keywords": matched_keywords

        })

    ranked_diseases.sort(
        key=lambda item: item["rerank_score"]
    )

    ranked_pairs = [
        (item["plant"], item["disease"])
        for item in ranked_diseases
    ]

    expected_plant_norm = normalize_text(expected_plant)
    expected_disease_norm = normalize_text(expected_disease)

    def pair_match(pair):

        return (
            normalize_text(pair[0]) == expected_plant_norm
            and normalize_text(pair[1]) == expected_disease_norm
        )

    top1_match = (
        len(ranked_pairs) > 0
        and pair_match(ranked_pairs[0])
    )

    top3_match = any(
        pair_match(pair)
        for pair in ranked_pairs[:3]
    )

    plant_top1_match = (
        len(ranked_pairs) > 0
        and normalize_text(ranked_pairs[0][0])
        == expected_plant_norm
    )

    plant_top3_match = any(
        normalize_text(pair[0]) == expected_plant_norm
        for pair in ranked_pairs[:3]
    )

    hit_at_1 += int(top1_match)
    hit_at_3 += int(top3_match)

    plant_hit_at_1 += int(plant_top1_match)
    plant_hit_at_3 += int(plant_top3_match)

    if ranked_diseases:

        top1 = ranked_diseases[0]

        top1_plant = top1["plant"]
        top1_disease = top1["disease"]
        top1_section = top1["section"]

        top1_distance = top1["distance"]
        top1_bonus = top1["keyword_bonus"]

        matched = ", ".join(top1["matched_keywords"])

    else:

        top1_plant = "None"
        top1_disease = "None"
        top1_section = "None"

        top1_distance = None
        top1_bonus = None

        matched = ""

    top3_display = " | ".join(
        f'{item["plant"]}: {item["disease"]}'
        for item in ranked_diseases[:3]
    )

    print(f"Question {index}: {question}")

    print(
        f"Expected: {expected_plant} - {expected_disease}"
    )

    print(
        f"Top 1: {top1_plant} - {top1_disease}"
    )

    print(
        f"Embedding distance: {top1_distance}"
    )

    print(
        f"Keyword bonus: {top1_bonus}"
    )

    print(
        f"Matched keywords: {matched}"
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

        "embedding_distance": top1_distance,

        "keyword_bonus": top1_bonus,

        "matched_keywords": matched,

        "top3_results": top3_display,

        "hit_at_1": int(top1_match),

        "hit_at_3": int(top3_match),

        "plant_hit_at_1": int(plant_top1_match),

        "plant_hit_at_3": int(plant_top3_match)

    })


# ==========================================
# 8. CALCULATE METRICS
# ==========================================

total_questions = len(test_cases)

hit1_rate = hit_at_1 / total_questions
hit3_rate = hit_at_3 / total_questions

plant_hit1_rate = plant_hit_at_1 / total_questions
plant_hit3_rate = plant_hit_at_3 / total_questions


print("\n========== RERANKING EVALUATION ==========")

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
# 9. SAVE RESULTS
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
            "embedding_distance",
            "keyword_bonus",
            "matched_keywords",
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

print("Reranking evaluation completed!")
