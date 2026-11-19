import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.rag_service import (
    load_knowledge_base,
    search_knowledge
)

from backend.services.llm_service import generate_answer


def main():

    question = (
        "Bệnh Early blight trên cà chua có triệu chứng gì "
        "và làm thế nào để phòng ngừa?"
    )

    print("========== PLANTCARE AI RAG + LLM TEST ==========")

    try:

        # STEP 1: Load knowledge base
        print("\n[1] Loading ChromaDB...")
        load_knowledge_base()

        # STEP 2: Retrieve relevant knowledge
        print("\n[2] Searching knowledge...")

        results = search_knowledge(
            question=question,
            plant="Tomato",
            top_k=8
        )

        if not results:
            print("No relevant knowledge found.")
            return

        print("Retrieved chunks:", len(results))

        # STEP 3: Build context from actual RAG results
        context_parts = []

        for index, item in enumerate(results, start=1):

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

        # STEP 4: Generate answer using Gemini
        print("\n[3] Generating answer with Gemini...")

        answer = generate_answer(
            question=question,
            context=context
        )

        # STEP 5: Display result
        print("\n========== AI ANSWER ==========")
        print(answer)

        print("\n========== RETRIEVED SOURCES ==========")

        for index, item in enumerate(results, start=1):

            print(f"\n{index}. {item.get('plant')} - {item.get('disease')}")
            print("Section:", item.get("section"))
            print("Source:", item.get("source"))

        print("\nRAG + LLM TEST: PASSED")

    except Exception as error:

        print("\nRAG + LLM TEST: FAILED")
        print("Error:", type(error).__name__)
        print("Message:", str(error))


if __name__ == "__main__":
    main()
