from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from pydantic import BaseModel, Field

from fastapi.middleware.cors import CORSMiddleware

from backend.services.prediction_service import (
    load_prediction_model,
    predict_image
)

from backend.services.rag_service import (
    load_knowledge_base,
    get_disease_knowledge,
    search_knowledge,
    detect_plant,
    detect_disease,
    detect_unsupported_plant
)
from backend.services.llm_service import generate_answer
# ==========================================
# 1. APPLICATION LIFESPAN
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting Plant Disease Diagnosis API...")

    # Load classification model once
    load_prediction_model()

    # Load RAG resources once
    load_knowledge_base()

    yield

    print("Shutting down API...")


# ==========================================
# 2. CREATE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Plant Disease Diagnosis API",
    description=(
        "Plant disease diagnosis using MobileNetV2 "
        "and knowledge retrieval using RAG."
    ),
    version="2.0.0",
    lifespan=lifespan
)


# ==========================================
# 3. CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==========================================
# 4. HEALTH CHECK API
# ==========================================

@app.get("/")
def root():

    return {
        "message": "Plant Disease Diagnosis API",
        "status": "running"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "MobileNetV2 Phase 2",
        "knowledge_base": "plant_disease_knowledge_chunks"
    }

# ==========================================
# KNOWLEDGE SEARCH API
# ==========================================

class KnowledgeSearchRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's question about plant diseases"
    )

    plant: Optional[str] = Field(
    default=None,
    description="Optional plant name"
    )

    top_k: int = Field(
        default=8,
        ge=1,
        le=20
    )
class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's question"
    )

    plant: Optional[str] = Field(
        default=None,
        description="Optional plant filter"
    )

    disease: Optional[str] = Field(
        default=None,
        description="Disease predicted by the image classification model"
    )

    top_k: int = Field(
        default=8,
        ge=1,
        le=20
    )
@app.post("/knowledge/search")
def knowledge_search(request: KnowledgeSearchRequest):

    try:

        results = search_knowledge(
            question=request.question,
            plant=request.plant,
            top_k=request.top_k
        )

        return {
            "success": True,
            "question": request.question,
            "plant_filter": request.plant,
            "total_results": len(results),
            "results": results
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("Knowledge search error:", error)

        raise HTTPException(
            status_code=500,
            detail="An error occurred during knowledge search."
        )


# ==========================================
# PLANTCARE AI CHAT API
# ==========================================


@app.post("/chat")
def chat(request: ChatRequest):

    try:

        # STEP 1: Identify plant and disease

        question_plant = detect_plant(request.question)

        # If the user explicitly mentions a plant,
        # prioritize it over the previous image prediction.
        detected_plant = question_plant or request.plant

        detected_disease = detect_disease(
            request.question,
            detected_plant
        )

        question_lower = request.question.lower()

        # STEP 1.1: Check unsupported plant

        unsupported_plant = detect_unsupported_plant(
            request.question
        )

        if unsupported_plant:

            return {
                "success": True,
                "question": request.question,
                "answer": (
                    f"Xin lỗi, hiện tại PlantCare AI chưa hỗ trợ "
                    f"tư vấn cho cây {unsupported_plant}. "
                    "Cơ sở dữ liệu hiện tại chỉ hỗ trợ 14 loại cây "
                    "đã được tích hợp. Bạn có thể thử hỏi về một "
                    "loại cây được hỗ trợ."
                ),
                "knowledge": {
                    "available": False,
                    "plant": unsupported_plant,
                    "disease": None,
                    "scope": "unsupported_plant",
                    "retrieved_chunks": 0,
                    "sources": []
                }
            }
        # Check whether the question is related
        # to agriculture, plants or plant diseases.

        agriculture_keywords = [
            # Vietnamese
            "cây", "lá", "rau", "quả", "trái",
            "vườn", "trồng", "canh tác", "nông nghiệp",
            "sâu", "nhện", "nấm", "virus", "vi khuẩn",
            "bệnh", "triệu chứng", "điều trị", "chữa",
            "thuốc", "phân bón", "tưới", "đất",
            "héo", "vàng lá", "đốm lá", "chăm sóc",
            "thu hoạch", "phòng ngừa", "phòng tránh",

            # English
            "plant", "crop", "leaf", "leaves",
            "disease", "pest", "fungus", "fungal",
            "virus", "bacteria", "agriculture",
            "farming", "cultivation", "fertilizer",
            "watering", "soil", "harvest",
            "treatment", "symptom", "care",
            "prevention", "tomato", "apple",
            "potato", "corn", "grape"
        ]

        agriculture_intent = any(
            keyword in question_lower
            for keyword in agriculture_keywords
        ) or question_plant is not None or detected_disease is not None

        # Reject questions outside the agricultural domain.
        if not agriculture_intent:
            return {
                "success": True,
                "question": request.question,
                "answer": (
                    "Chào bạn, tôi là PlantCare AI, một trợ lý chuyên "
                    "tư vấn về bệnh cây trồng và chăm sóc cây trồng. "
                    "Rất tiếc, câu hỏi này nằm ngoài phạm vi chuyên môn "
                    "của tôi nên tôi không thể cung cấp thông tin. "
                    "Bạn có thể hỏi tôi về bệnh cây trồng, triệu chứng, "
                    "cách điều trị hoặc phương pháp chăm sóc cây nhé!"
                ),
                "knowledge": {
                    "available": False,
                    "plant": None,
                    "disease": None,
                    "scope": "out_of_scope",
                    "retrieved_chunks": 0,
                    "sources": []
                }
            }
        # STEP 2: Detect user intent

        healthy_keywords = [
            "khỏe mạnh",
            "cây khỏe",
            "cây đang khỏe",
            "healthy"
        ]

        healthy_intent = any(
            keyword in question_lower
            for keyword in healthy_keywords
        )

        contextual_keywords = [
            "cây của tôi",
            "cây này",
            "bệnh này",
            "bệnh vừa",
            "kết quả chẩn đoán",
            "chẩn đoán",
            "điều trị",
            "chữa bệnh",
            "phòng tránh",
            "phòng ngừa",
            "tái phát",
            "mắc bệnh gì",
            "bệnh gì",
            "triệu chứng"
        ]

        contextual_intent = any(
            keyword in question_lower
            for keyword in contextual_keywords
        )
        # Initialize response disease before retrieval
        response_disease = detected_disease
        # STEP 3: Retrieve the correct knowledge

        if healthy_intent:

            # Healthy plant questions
            results = search_knowledge(
                question=request.question,
                plant=detected_plant,
                top_k=request.top_k
            )

            relevant_results = [
                item for item in results
                if item.get("disease", "").lower() == "healthy"
            ]

        elif detected_disease:

            # Explicit disease mentioned by the user
            relevant_results = get_disease_knowledge(
                detected_plant,
                detected_disease
            ) if detected_plant else []

        elif request.disease and contextual_intent:

            # Use the disease predicted from the uploaded image
            relevant_results = get_disease_knowledge(
                detected_plant,
                request.disease
            ) if detected_plant else []

        else:

            # General plant-care questions
            relevant_results = search_knowledge(
                question=request.question,
                plant=detected_plant,
                top_k=request.top_k
            )
            # STEP 3.1: Determine the disease used in retrieval

        if (
            not response_disease
            and request.disease
            and contextual_intent
            and not healthy_intent
        ):
            response_disease = request.disease

        # STEP 4: Build context for Gemini
        context_parts = []

        for index, item in enumerate(
            relevant_results,
            start=1
        ):
            context_parts.append(
                f"""
Knowledge {index}
Plant: {item.get('plant', '')}
Disease: {item.get('disease', '')}
Section: {item.get('section', '')}
Content: {item.get('content', '')}
Source: {item.get('source', '')}
"""
            )

        context = "\n".join(context_parts)

        if not context:
            context = (
                "Không tìm thấy kiến thức phù hợp "
                "trong cơ sở dữ liệu PlantCare AI."
            )

        # STEP 5: Generate answer using Gemini
        answer = generate_answer(
            question=request.question,
            context=context
        )

        # STEP 6: Collect unique sources
        sources = []
        seen_sources = set()

        for item in relevant_results:
            source = item.get("source", "")

            source_key = (
                item.get("plant"),
                item.get("disease"),
                source
            )

            if source and source_key not in seen_sources:
                seen_sources.add(source_key)

                sources.append({
                    "plant": item.get("plant"),
                    "disease": item.get("disease"),
                    "source": source
                })

        # STEP 7: Return response
        return {
            "success": True,
            "question": request.question,
            "answer": answer,
            "knowledge": {
                "available": len(relevant_results) > 0,
                "plant": (
                    detected_plant
                    or (
                        relevant_results[0].get("plant")
                        if relevant_results else None
                    )
                ),
                "disease": response_disease,
                "scope": (
                "specific_disease"
                if response_disease
                else "general_plant_knowledge"
                ),
                "retrieved_chunks": len(relevant_results),
                "sources": sources
            }
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("Chat API error:", error)

        raise HTTPException(
            status_code=500,
            detail="An error occurred during AI chat."
        )
# ==========================================
# 5. IMAGE PREDICTION API
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    filename = file.filename or ""

    extension = (
        "." + filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG and WEBP images are supported."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MB."
        )

    try:

        # Step 1: Predict plant disease
        prediction = predict_image(image_bytes)

        top_prediction = prediction["top_prediction"]

        plant = top_prediction["plant"]
        disease = top_prediction["disease"]

        # Step 2: Normalize disease name
        disease_aliases = {
            "Cedar apple rust": "Cedar-apple rust"
        }

        knowledge_disease = disease_aliases.get(
            disease,
            disease
        )

        # Step 3: Retrieve related knowledge
        knowledge = get_disease_knowledge(
            plant,
            knowledge_disease
        )

        # Step 4: Prepare knowledge response
        knowledge_response = {
            "available": len(knowledge) > 0,
            "plant": plant,
            "disease": knowledge_disease,
            "message": (
                "Relevant agricultural knowledge retrieved successfully."
                if knowledge
                else "Knowledge for this plant or disease is not available yet."
            ),
            "sections": [
                {
                    "section": item["section"],
                    "content": item["content"],
                    "source": item["source"],
                    "source_file": item["source_file"]
                }
                for item in knowledge
            ]
        }

        # Step 5: Return combined result
        return {
            "success": True,
            "message": "Prediction and knowledge retrieval completed.",
            "data": {
                "prediction": prediction,
                "knowledge": knowledge_response
            }
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print("Prediction or RAG error:", error)

        raise HTTPException(
            status_code=500,
            detail="An error occurred during prediction or knowledge retrieval."
        )
