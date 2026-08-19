import os
import re
import base64
import asyncio
from io import BytesIO
from PIL import Image
from openai import OpenAI
import google.generativeai as genai  # pip install google-generativeai

# --- DeepSeek for rewriting ---
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# --- Gemini for Vision ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # fast and free

# --- OpenAI for DALL-E ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ... (clean_mentions, clean_text, rewrite_text remain the same) ...

async def describe_image_bytes(image_bytes: bytes) -> str:
    """Use Gemini Vision to describe image without watermarks."""
    try:
        img = Image.open(BytesIO(image_bytes))
        if getattr(img, 'is_animated', False):
            img.seek(0)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_data = buffered.getvalue()

        # Gemini expects a list of parts (text + image)
        prompt = """
        Describe this image concisely. Keep all sports details (teams, odds, scores).
        Ignore any usernames (like @cappersfree) and logos (like CF).
        Describe layout and style. Output a clear, short description.
        """
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            gemini_model.generate_content,
            [prompt, img_data]
        )
        description = response.text.strip()
        print(f"Gemini description: {description[:200]}...")
        return description
    except Exception as e:
        print(f"Gemini Vision error: {e}")
        return None
