


# Gemini API integration using google-genai
import google.genai as genai

GEMINI_API_KEY = "AIzaSyAorbMtZpb-kNN_kl4YReEwJKoymV0Kuns"

def get_response(user_query: str) -> str:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_query,
        )
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        return f"Error: {e}"
