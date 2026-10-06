import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Official Gemma 4 models supported on Gemini API
MODEL_GEMMA_4_26B = "gemma-4-26b-a4b-it"  # Fast, highly available MoE architecture
MODEL_GEMMA_4_31B = "gemma-4-31b-it"      # Dense reasoning model

DEFAULT_MODEL = MODEL_GEMMA_4_26B

SUPPORTED_MODELS = [
    MODEL_GEMMA_4_26B,
    MODEL_GEMMA_4_31B,
]

DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.95

def get_api_key(override_key: Optional[str] = None) -> Optional[str]:
    """Retrieve API key from explicit override or environment."""
    if override_key and override_key.strip():
        return override_key.strip()
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

def create_client(api_key: Optional[str] = None):
    """
    Initialize Google GenAI client.
    Returns None if no API key is available.
    """
    resolved_key = get_api_key(api_key)
    if not resolved_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=resolved_key)
    except Exception as exc:
        print(f"Warning: Failed to initialize genai.Client: {exc}")
        return None
