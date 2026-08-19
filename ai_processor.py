import os
import re
import base64
import requests
from io import BytesIO
from PIL import Image
from openai import OpenAI

# --- Clients ---
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# --- Config ---
NEW_MENTION = os.getenv("NEW_MENTION", "")  # e.g., "@MyNewChannel" – if empty, we just remove old mentions
OLD_MENTION = "@cappersfree"

def clean_mentions(text: str) -> str:
    """Remove or replace the old mention."""
    if not text:
        return text
    if NEW_MENTION:
        # Replace old mention with new one (case-insensitive)
        text = re.sub(OLD_MENTION, NEW_MENTION, text, flags=re.IGNORECASE)
    else:
        # Remove all @mentions (including the old one)
        text = re.sub(r'@\w+', '', text)
    return text

def clean_text(text: str) -> str:
    """Remove extra formatting and mentions."""
    if not text:
        return text
    # Remove @mentions
    text = clean_mentions(text)
    # Remove **bold** and __italic__ markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def rewrite_text(original_text: str) -> str:
    """Rewrite text with DeepSeek, then clean mentions/formatting."""
    cleaned = clean_text(original_text)
    if not cleaned or len(cleaned) < 5:
        return original_text
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Rewrite the following text to make it unique while preserving the core meaning. Return only the rewritten text, without any markdown, bold, italics, or extra formatting. Keep it plain."},
                {"role": "user", "content": cleaned}
            ],
            temperature=0.8,
            max_tokens=2000
        )
        rewritten = response.choices[0].message.content.strip()
        # Final cleaning
        rewritten = clean_mentions(rewritten)
        rewritten = re.sub(r'\*\*([^*]+)\*\*', r'\1', rewritten)
        return rewritten
    except Exception as e:
        print(f"DeepSeek error: {e}")
        return original_text

async def describe_image_for_regeneration(image_url: str) -> str:
    """
    Use GPT-4 Vision to get a clean description of the image,
    ignoring any watermarks, logos, or text.
    """
    try:
        # Download the image (or GIF) and extract first frame if needed
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            raise Exception("Failed to download image")

        img = Image.open(BytesIO(response.content))
        # If it's a GIF, get the first frame
        if getattr(img, 'is_animated', False):
            img.seek(0)  # first frame
        # Convert to RGB and encode as base64 JPEG
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{img_base64}"

        # Call GPT-4 Vision
        vision_response = openai_client.chat.completions.create(
            model="gpt-4-turbo",  # or "gpt-4-vision-preview" if still available
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail. Ignore any watermarks, logos, text overlays, or branding. Focus on the main subject, colors, composition, action, and mood. Do not mention any text you see. Keep the description concise but vivid, suitable as a prompt for an image generation model."},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ],
            max_tokens=300
        )
        description = vision_response.choices[0].message.content.strip()
        print(f"Vision description: {description[:100]}...")
        return description
    except Exception as e:
        print(f"Vision error: {e}")
        return None

async def generate_image_from_description(prompt: str) -> str:
    """Generate a new image using DALL-E 3 from the description."""
    if not prompt or len(prompt) < 10:
        return None
    try:
        # Shorten prompt if needed
        safe_prompt = f"A professional illustration representing: {prompt[:300]}"
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=safe_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        print(f"DALL-E error: {e}")
        return None

async def regenerate_image_from_url(image_url: str) -> str:
    """Full pipeline: describe image -> generate new image."""
    description = await describe_image_for_regeneration(image_url)
    if not description:
        return None
    new_url = await generate_image_from_description(description)
    return new_url
