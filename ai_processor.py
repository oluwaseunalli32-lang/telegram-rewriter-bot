import os
from openai import OpenAI

# DeepSeek client (uses OpenAI compatible base URL)
deepseek_client = OpenAI(
    api_key=os.getenv("sk-2404532f781f443a9c30ad807dd14b34"),
    base_url="https://api.deepseek.com/v1"
)

# OpenAI client for DALL-E
openai_client = OpenAI(
    api_key=os.getenv("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")
)

async def rewrite_text(original_text: str) -> str:
    """Send original text to DeepSeek and get a unique rewrite."""
    if not original_text or len(original_text.strip()) < 5:
        return original_text
    
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a content rewriter. Rewrite the following text to make it completely unique while preserving all key information, tone, and meaning. Keep the same length. Only output the rewritten text, nothing else."},
                {"role": "user", "content": original_text}
            ],
            temperature=0.8,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"DeepSeek error: {e}")
        return original_text  # Fallback to original

async def generate_image(prompt: str) -> str:
    """Generate an image using OpenAI DALL-E 3 and return the URL."""
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=f"Create a professional, visually striking cover image for this content: {prompt[:200]}",
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        print(f"OpenAI image error: {e}")
        return None  # No image if it fails