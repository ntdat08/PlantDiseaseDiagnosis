# PlantCare AI – Plant Disease Diagnosis and Cultivation Advisory System

## 1. Introduction

PlantCare AI is an AI-powered web application designed to identify plant diseases from leaf images and provide relevant disease information and cultivation advice.

The system combines a deep learning model for plant disease classification, a Retrieval-Augmented Generation (RAG) pipeline for retrieving agricultural knowledge, and Gemini AI for conversational agricultural consultation.

## 2. Main Features

- **Plant Disease Detection:** Identify plant diseases from uploaded leaf images.
- **Top-3 Predictions:** Display the three most likely prediction results with confidence scores.
- **Disease Knowledge Retrieval:** Retrieve information about symptoms, causes, prevention, and treatment.
- **AI Agricultural Chatbot:** Answer plant-related questions using Gemini AI and retrieved knowledge.
- **Interactive Web Interface:** Upload images, view prediction results, and communicate with the chatbot.

## 3. Technologies Used

### Backend and AI
- Python 3.9
- TensorFlow / Keras
- MobileNetV2
- FastAPI
- Sentence Transformers
- ChromaDB
- Gemini API

### Frontend
- React
- Vite
- JavaScript
- CSS
- Axios
- Lucide React

### Knowledge Retrieval
- Embedding Model: `paraphrase-multilingual-MiniLM-L12-v2`
- Vector Database: ChromaDB
- Retrieval-Augmented Generation (RAG)

## 4. AI Model

The plant disease classification model is based on MobileNetV2 with ImageNet pretrained weights and fine-tuning.

| Configuration | Value |
|---|---|
| Model | MobileNetV2 |
| Input Image Size | 224 × 224 |
| Number of Classes | 38 |
| Number of Plant Types | 14 |
| Output | Top-3 predictions |

### Model Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 97.08% |
| Macro Precision | 96.31% |
| Macro Recall | 96.08% |
| Macro F1-score | 96.11% |
| Weighted F1-score | 97.07% |

*Note: These results were obtained from the project's test dataset. Performance on real-world field images may differ.*

## 5. Project Structure

```text
PlantDiseaseDiagnosis/
├── backend/
│   ├── api/
│   └── services/
├── frontend/
│   ├── src/
│   └── package.json
├── knowledge_base/
│   ├── documents/
│   │   └── diseases/
│   └── processed/
├── dataset/
│   └── test_images/
├── models/
│   └── mobilenetv2_best_phase2.h5
├── scripts/
│   └── build_knowledge_base_chunks.py
├── evaluation/
├── vector_store/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 6. Installation and Setup

### Prerequisites

- Python 3.9
- Node.js and npm
- Git
- Gemini API key

### Step 1: Clone the Repository

```bash
git clone https://github.com/ntdat08/PlantDiseaseDiagnosis.git
cd PlantDiseaseDiagnosis
```

### Step 2: Create a Python Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install Python Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Copy `.env.example` to `.env`.

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

Open `.env` and configure your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Obtain an API key from Google AI Studio:

https://aistudio.google.com/apikey

**Important:** Never commit your `.env` file or expose your API key publicly.

### Step 5: Prepare the Trained Model

The trained model weights are not included in this repository.

Obtain the trained model file:

`mobilenetv2_best_phase2.h5`

Place it in the following directory:

```text
models/mobilenetv2_best_phase2.h5
```

The backend requires this file to perform plant disease predictions.

### Step 6: Build the Knowledge Base

The ChromaDB vector database is generated locally and is not included in this repository.

Run:

```bash
python scripts/build_knowledge_base_chunks.py
```

This script:
- Loads and validates 38 disease knowledge records.
- Creates 152 knowledge chunks.
- Generates 384-dimensional embeddings.
- Builds the ChromaDB collection.

The generated database is saved in:

```text
vector_store/
```

## 7. Running the Application

### Start the Backend

Open the first terminal, activate the virtual environment, and run:

```bash
python -m uvicorn backend.api.main:app --reload
```

Backend API:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

### Start the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the application:

http://localhost:5173

## 8. System Workflow

1. The user uploads a plant leaf image.
2. MobileNetV2 analyzes the image and predicts the most likely plant disease.
3. The RAG pipeline retrieves relevant agricultural knowledge from ChromaDB.
4. The user can ask additional questions through the AI chatbot.
5. Gemini AI generates an answer using the retrieved knowledge and available prediction context.

## 9. Limitations

- The classification model was evaluated using the PlantVillage dataset, so real-world performance may vary.
- The chatbot requires a valid Gemini API key and is subject to API quotas and rate limits.
- Predictions are AI-generated suggestions and should not be treated as definitive agricultural diagnoses.

## 10. Project Status

The main application features have been implemented and tested, including:

- Plant disease classification.
- Knowledge base construction and retrieval.
- Gemini-powered agricultural chatbot.
- FastAPI backend.
- React web interface.

**Project Status: Core Features Completed**