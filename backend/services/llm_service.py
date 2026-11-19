import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

_client = None


# ==========================================
# 2. INITIALIZE GEMINI CLIENT
# ==========================================

def get_gemini_client():

    global _client

    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY was not found in the .env file."
        )

    if _client is None:
        _client = genai.Client(api_key=API_KEY)

    return _client


# ==========================================
# 3. GENERATE AI ANSWER
# ==========================================

def generate_answer(question: str, context: str) -> str:

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context or not context.strip():
        context = (
            "Không tìm thấy kiến thức phù hợp trong cơ sở dữ liệu."
        )

    prompt = f"""
Bạn là PlantCare AI, một trợ lý AI chuyên tư vấn
về bệnh cây trồng và chăm sóc cây trồng.

NHIỆM VỤ:
Trả lời câu hỏi của người dùng bằng tiếng Việt,
dễ hiểu, chính xác và có tính ứng dụng thực tế.

NGUYÊN TẮC:
1. Ưu tiên sử dụng thông tin trong KNOWLEDGE CONTEXT.
2. Không tự bịa tên bệnh, nguyên nhân hoặc phương pháp điều trị.
3. Không khẳng định chắc chắn nếu dữ liệu chưa đủ.
4. Nếu kiến thức được cung cấp không đủ để trả lời,
   hãy nói rõ điều đó.
5. Đưa ra hướng dẫn ngắn gọn, chia thành các ý dễ đọc.
6. Không đưa ra khuyến nghị sử dụng thuốc bảo vệ thực vật
   với liều lượng cụ thể nếu nguồn không cung cấp.
7. Phân biệt rõ thông tin từ cơ sở dữ liệu và lời khuyên tổng quát.
8. Nếu KNOWLEDGE CONTEXT chứa thông tin về một bệnh
   cụ thể và người dùng hỏi về "cây của tôi",
   "bệnh này", "cây vừa chẩn đoán", hãy sử dụng
   bệnh đó làm ngữ cảnh tư vấn.

9. Không được nói người dùng chưa cung cấp hình ảnh
   hoặc chưa nêu tên cây nếu thông tin đó đã có
   trong KNOWLEDGE CONTEXT.

10. Khi sử dụng kết quả dự đoán của mô hình,
    phải nói rõ đây là kết quả AI dự đoán,
    không khẳng định đó là chẩn đoán chắc chắn.

11. Nếu người dùng hỏi rõ một bệnh khác,
    chỉ tập trung trả lời bệnh được hỏi,
    không trộn lẫn kiến thức của các bệnh khác.

12. Không tự tạo ra phần trăm độ tin cậy,
    liều lượng thuốc hoặc số liệu không có
    trong dữ liệu được cung cấp.
KNOWLEDGE CONTEXT:
{context}

USER QUESTION:
{question}

Hãy đưa ra câu trả lời trực tiếp, hữu ích và bằng tiếng Việt.
"""

    client = get_gemini_client()

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1000
        )
    )

    answer = response.text

    if not answer or not answer.strip():
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return answer.strip()
