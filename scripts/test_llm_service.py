import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.llm_service import generate_answer


def main():

    question = (
        "Bệnh Early blight trên cà chua là gì? "
        "Làm thế nào để phòng ngừa bệnh này?"
    )

    context = """
Plant: Tomato
Disease: Early blight

Symptoms:
Bệnh gây ra các đốm nâu trên lá, thường có
các vòng đồng tâm giống hình bia bắn.

Causes:
Bệnh do nấm Alternaria solani gây ra.

Prevention:
Luân canh cây trồng, vệ sinh tàn dư thực vật,
hạn chế làm ướt lá khi tưới.

Treatment:
Loại bỏ lá bị nhiễm bệnh và áp dụng biện pháp
quản lý bệnh phù hợp.
"""

    print("Testing PlantCare AI LLM Service...")

    try:

        answer = generate_answer(
            question=question,
            context=context
        )

        print("\n========== AI ANSWER ==========")
        print(answer)
        print("==============================")

        print("\nLLM SERVICE TEST: PASSED")

    except Exception as error:

        print("\nLLM SERVICE TEST: FAILED")
        print("Error:", type(error).__name__)
        print("Message:", str(error))


if __name__ == "__main__":
    main()
