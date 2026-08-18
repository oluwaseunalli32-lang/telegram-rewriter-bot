import os
import re
from openai import OpenAI

# DeepSeek client (text rewrite)
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# OpenAI client (DALL‑E image generation)
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def clean_text(text: str) -> str:
    """Remove @mentions, **bold**, extra spaces, and common Telegram formatting."""
    if not text:
        return text
    # Remove @mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)
    # Remove **bold** and __italic__ markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def rewrite_text(original_text: str) -> str:
    """Rewrite the text using DeepSeek, with pre‑cleaning."""
    # 1. Clean the original
    cleaned = clean_text(original_text)
    if not cleaned or len(cleaned) < 5:
        return original_text

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Rewrite the following text to make it unique while preserving the core meaning. Return only the rewritten text, without any markdown, bold, italics, or extra formatting. Keep it clean and plain."},
                {"role": "user", "content": cleaned}
            ],
            temperature=0.8,
            max_tokens=2000
        )
        rewritten = response.choices[0].message.content.strip()
        # Remove any lingering markdown that DeepSeek might add
        rewritten = re.sub(r'\*\*([^*]+)\*\*', r'\1', rewritten)
        return rewritten
    except Exception as e:
        print(f"DeepSeek error: {e}")
        return original_text

async def generate_image(prompt: str) -> str:
    """Generate an image using DALL‑E 3, log the error if it fails."""
    if not prompt or len(prompt.strip()) < 10:
        print("Prompt too short for image generation")
        return None
    try:
        # Use a shorter, safer prompt
        safe_prompt = f"A professional abstract illustration representing: {prompt[:150]}"
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=safe_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        # Log the actual error to help debug
        print(f"OpenAI image error: {e}")
        return None
