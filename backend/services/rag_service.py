from pathlib import Path
import re
import unicodedata

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_DB_PATH = PROJECT_ROOT / "vector_store"

COLLECTION_NAME = "plant_disease_knowledge_chunks"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

DEFAULT_TOP_K = 8

SECTION_ORDER = [
    "symptoms",
    "causes",
    "prevention",
    "treatment"
]

# Trọng số ưu tiên khi reranking
DISEASE_BONUS = 0.25
SECTION_BONUS = 0.20


# ==========================================
# 2. GLOBAL RESOURCES
# ==========================================

embedding_model = None
chroma_client = None
collection = None


# ==========================================
# 3. TEXT NORMALIZATION
# ==========================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text).lower().strip()

    text = text.replace("đ", "d")

    text = unicodedata.normalize("NFD", text)

    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )

    text = re.sub(r"[^a-z0-9]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


# ==========================================
# 4. LOAD KNOWLEDGE RESOURCES
# ==========================================

def load_knowledge_base():

    global embedding_model
    global chroma_client
    global collection

    if (
        embedding_model is not None
        and collection is not None
    ):
        return

    if not VECTOR_DB_PATH.exists():
        raise FileNotFoundError(
            f"Vector database not found: {VECTOR_DB_PATH}"
        )

    print("Loading knowledge embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )

    print("Connecting to ChromaDB...")

    chroma_client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH)
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    if collection.count() == 0:
        raise ValueError(
            "Knowledge base collection is empty."
        )

    print("Knowledge base loaded successfully!")
    print("Collection:", collection.name)
    print("Total chunks:", collection.count())


# ==========================================
# 5. GET KNOWLEDGE FOR A SPECIFIC DISEASE
# ==========================================

def get_disease_knowledge(plant, disease):

    load_knowledge_base()

    results = collection.get(
        where={
            "$and": [
                {"plant": plant},
                {"disease": disease}
            ]
        },
        include=[
            "documents",
            "metadatas"
        ]
    )

    knowledge = []

    for i in range(len(results["ids"])):

        metadata = results["metadatas"][i]

        knowledge.append({
            "id": results["ids"][i],
            "plant": metadata["plant"],
            "disease": metadata["disease"],
            "section": metadata["section"],
            "content": results["documents"][i],
            "source": metadata.get("source", ""),
            "source_file": metadata.get("source_file", "")
        })

    knowledge.sort(
        key=lambda item: (
            SECTION_ORDER.index(item["section"])
            if item["section"] in SECTION_ORDER
            else len(SECTION_ORDER)
        )
    )

    return knowledge


# ==========================================
# 6. DETECT PLANT
# ==========================================

PLANT_ALIASES = {
    "Apple": [
        "apple",
        "tao",
        "cay tao"
    ],
    "Blueberry": [
        "blueberry",
        "viet quat"
    ],
    "Cherry (including sour)": [
        "cherry",
        "anh dao"
    ],
    "Corn (maize)": [
        "corn",
        "maize",
        "ngo",
        "bap"
    ],
    "Grape": [
        "grape",
        "nho",
        "cay nho"
    ],
    "Orange": [
        "orange",
        "cam"
    ],
    "Peach": [
        "peach",
        "dao"
    ],
    "Pepper bell": [
        "pepper",
        "bell pepper",
        "ot chuong"
    ],
    "Potato": [
        "potato",
        "khoai tay"
    ],
    "Raspberry": [
        "raspberry",
        "mam xoi"
    ],
    "Soybean": [
        "soybean",
        "soy",
        "dau nanh"
    ],
    "Squash": [
        "squash",
        "bi",
        "bi ngo"
    ],
    "Strawberry": [
        "strawberry",
        "dau tay"
    ],
    "Tomato": [
        "tomato",
        "ca chua"
    ]
}


def detect_plant(question):

    normalized_question = normalize_text(question)

    matches = []

    for plant, aliases in PLANT_ALIASES.items():

        for alias in aliases:

            normalized_alias = normalize_text(alias)

            pattern = (
                r"\b"
                + re.escape(normalized_alias)
                + r"\b"
            )

            if re.search(pattern, normalized_question):

                matches.append(
                    (len(normalized_alias), plant)
                )

    if not matches:
        return None

    matches.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return matches[0][1]


# ==========================================
# 6.1 DETECT UNSUPPORTED PLANTS
# ==========================================

UNSUPPORTED_PLANT_ALIASES = {
    "Hoa hồng": [
        "hoa hong",
        "rose",
        "roses"
    ]
}


def detect_unsupported_plant(question):

    normalized_question = normalize_text(question)

    matches = []

    for plant, aliases in UNSUPPORTED_PLANT_ALIASES.items():

        for alias in aliases:

            normalized_alias = normalize_text(alias)

            pattern = (
                r"\b"
                + re.escape(normalized_alias)
                + r"\b"
            )

            if re.search(pattern, normalized_question):

                matches.append(
                    (len(normalized_alias), plant)
                )

    if not matches:
        return None

    matches.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return matches[0][1]

# ==========================================
# 7. DETECT QUESTION SECTION
# ==========================================

SECTION_KEYWORDS = {
    "symptoms": [
        "symptom",
        "symptoms",
        "dau hieu",
        "bieu hien",
        "trieu chung",
        "nhan biet",
        "nhan dang"
    ],
    "causes": [
        "cause",
        "causes",
        "nguyen nhan",
        "tai sao",
        "vi sao",
        "nguon goc"
    ],
    "prevention": [
        "prevention",
        "prevent",
        "phong ngua",
        "phong benh",
        "ngan ngua",
        "phong tranh",
        "cach phong"
    ],
    "treatment": [
        "treatment",
        "treat",
        "cure",
        "dieu tri",
        "chua benh",
        "khac phuc",
        "xu ly"
    ]
}


def detect_section(question):

    normalized_question = normalize_text(question)

    for section, keywords in SECTION_KEYWORDS.items():

        for keyword in keywords:

            normalized_keyword = normalize_text(keyword)

            pattern = (
                r"\b"
                + re.escape(normalized_keyword)
                + r"\b"
            )

            if re.search(pattern, normalized_question):
                return section

    return None


# ==========================================
# 8. DETECT DISEASE
# ==========================================

DISEASE_ALIASES = {
    "Early blight": [
        "benh dom nau",
        "dom nau",
        "benh dom vong som",
        "dom vong som"
    ],
    "Late blight": [
        "benh moc suong",
        "moc suong"
    ],
    "Bacterial spot": [
        "benh dom vi khuan",
        "dom vi khuan"
    ],
    "Powdery mildew": [
        "benh phan trang",
        "phan trang"
    ],
    "Black rot": [
        "benh thoi den",
        "thoi den"
    ],
    "Apple scab": [
        "benh ghe tao",
        "benh dom ghe tao"
    ],
    "Cedar-apple rust": [
        "benh ri sat tao",
        "ri sat tao"
    ],
    "Common rust": [
        "benh ri sat thong thuong",
        "ri sat ngo"
    ],
    "Leaf Mold": [
        "benh moc la",
        "moc la"
    ],
    "Target Spot": [
        "benh dom vong",
        "dom vong"
    ],
    "Spider mites Two-spotted spider mite": [
        "nhen do",
        "nhen hai cham",
        "nhen hai cham do"
    ],
    "Tomato Yellow Leaf Curl Virus": [
        "benh xoan la vang ca chua",
        "xoan la vang"
    ],
    "Tomato mosaic virus": [
        "virus kham ca chua",
        "benh kham la ca chua"
    ],
    "Citrus greening": [
        "benh vang la greeening",
        "benh vang la gan xanh"
    ]
}


def detect_disease(question, plant=None):

    normalized_question = normalize_text(question)

    disease_alias_matches = []

    for disease, aliases in DISEASE_ALIASES.items():

        for alias in aliases:

            normalized_alias = normalize_text(alias)

            pattern = (
                r"\b"
                + re.escape(normalized_alias)
                + r"\b"
            )

            if re.search(pattern, normalized_question):

                disease_alias_matches.append(
                    (len(normalized_alias), disease)
                )

    if disease_alias_matches:

        disease_alias_matches.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return disease_alias_matches[0][1]

    # Tìm tên bệnh tiếng Anh trực tiếp trong câu hỏi
    results = collection.get(
        include=["metadatas"]
    )

    diseases = set()

    for metadata in results["metadatas"]:

        if plant and metadata["plant"] != plant:
            continue

        diseases.add(metadata["disease"])

    disease_matches = []

    for disease in diseases:

        normalized_disease = normalize_text(disease)

        if not normalized_disease:
            continue

        pattern = (
            r"\b"
            + re.escape(normalized_disease)
            + r"\b"
        )

        if re.search(pattern, normalized_question):

            disease_matches.append(
                (len(normalized_disease), disease)
            )

    if not disease_matches:
        return None

    disease_matches.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return disease_matches[0][1]


# ==========================================
# 9. SEMANTIC RETRIEVAL + RERANKING
# ==========================================

def search_knowledge(
    question,
    plant=None,
    top_k=DEFAULT_TOP_K
):

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(
            "top_k must be a positive integer."
        )

    load_knowledge_base()

    # --------------------------------------
    # 9.1 Detect information from question
    # --------------------------------------

    detected_plant = plant or detect_plant(question)

    detected_disease = detect_disease(
        question,
        detected_plant
    )

    detected_section = detect_section(question)

    # --------------------------------------
    # 9.2 Create query embedding
    # --------------------------------------

    query_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # --------------------------------------
    # 9.3 Retrieve candidate chunks
    # --------------------------------------

    query_options = {}

    if detected_plant:
        query_options["where"] = {
            "plant": detected_plant
        }

    candidate_count = collection.count()

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=candidate_count,
        include=[
            "documents",
            "metadatas",
            "distances"
        ],
        **query_options
    )

    # --------------------------------------
    # 9.4 Rerank retrieved chunks
    # --------------------------------------

    retrieved_chunks = []

    for i in range(len(results["ids"][0])):

        metadata = results["metadatas"][0][i]

        distance = float(
            results["distances"][0][i]
        )

        rerank_score = distance

        # Ưu tiên đúng bệnh
        if (
            detected_disease
            and normalize_text(metadata["disease"])
            == normalize_text(detected_disease)
        ):
            rerank_score -= DISEASE_BONUS

        # Ưu tiên đúng loại kiến thức
        if (
            detected_section
            and metadata["section"] == detected_section
        ):
            rerank_score -= SECTION_BONUS

        retrieved_chunks.append({
            "id": results["ids"][0][i],
            "plant": metadata["plant"],
            "disease": metadata["disease"],
            "section": metadata["section"],
            "content": results["documents"][0][i],
            "distance": round(distance, 4),
            "source": metadata.get("source", ""),
            "source_file": metadata.get("source_file", ""),
            "_rerank_score": rerank_score
        })

    # --------------------------------------
    # 9.5 Sort by reranking score
    # --------------------------------------

    retrieved_chunks.sort(
        key=lambda item: item["_rerank_score"]
    )

    # --------------------------------------
    # 9.6 Return Top-K
    # --------------------------------------

    final_results = []

    for item in retrieved_chunks[:top_k]:

        item.pop("_rerank_score", None)

        final_results.append(item)

    return final_results