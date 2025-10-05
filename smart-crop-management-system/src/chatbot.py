


# Gemini API integration using google-genai
import os
from google import genai

GEMINI_API_KEY = "AIzaSyAorbMtZpb-kNN_kl4YReEwJKoymV0Kuns"

def get_response(user_query: str) -> str:
    try:
        os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=user_query
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"
