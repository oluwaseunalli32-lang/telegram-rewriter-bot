import os
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

async def rewrite_text(original_text: str) -> str:
    """Rewrite the text using DeepSeek."""
    if not original_text or len(original_text.strip()) < 5:
        return original_text
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Rewrite the following text to make it unique while preserving the meaning and tone. Keep the same length."},
                {"role": "user", "content": original_text}
            ],
            temperature=0.8,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"DeepSeek error: {e}")
        return original_text  # fallback to original

async def generate_image(prompt: str) -> str:
    """Generate an image using DALL‑E 3, return the URL or None on failure."""
    if not prompt or len(prompt.strip()) < 10:
        return None
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=f"Create a professional, visually striking cover image for: {prompt[:200]}",
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        print(f"OpenAI image error: {e}")
        return None
