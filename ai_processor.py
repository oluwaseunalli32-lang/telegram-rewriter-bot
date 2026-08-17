import os
from openai import OpenAI

# DeepSeek client (uses OpenAI compatible base URL)
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # <-- Correct: reads from environment
    base_url="https://api.deepseek.com/v1"
)

# OpenAI client for DALL-E
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")     # <-- Correct: reads from environment
)

async def rewrite_text(original_text: str) -> str:
    # ... rest of the function (keep as before)

async def generate_image(prompt: str) -> str:
    # ... rest of the function (keep as before)
