import os

from dotenv import load_dotenv
from google import genai


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")


# ==========================================
# 2. TEST GEMINI CONNECTION
# ==========================================

def test_gemini_connection():

    if not API_KEY:
        print("ERROR: GEMINI_API_KEY not found in .env")
        return

    print("Connecting to Gemini API...")
    print("Model:", MODEL)

    try:

        client = genai.Client(api_key=API_KEY)

        response = client.models.generate_content(
            model=MODEL,
            contents="Reply exactly: PlantCare AI Gemini connection successful."
        )

        print("\n========== API TEST RESULT ==========")
        print("Status: SUCCESS")
        print("Response:", response.text)
        print("=====================================")

    except Exception as error:

        print("\n========== API TEST FAILED ==========")
        print("Error type:", type(error).__name__)
        print("Error message:", str(error))
        print("=====================================")


if __name__ == "__main__":
    test_gemini_connection()
