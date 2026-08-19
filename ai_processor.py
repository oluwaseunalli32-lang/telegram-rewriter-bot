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
NEW_MENTION = os.getenv("NEW_MENTION", "")
OLD_MENTION = "@cappersfree"

def clean_mentions(text: str) -> str:
    if not text:
        return text
    if NEW_MENTION:
        text = re.sub(OLD_MENTION, NEW_MENTION, text, flags=re.IGNORECASE)
    else:
        text = re.sub(r'@\w+', '', text)
    return text

def clean_text(text: str) -> str:
    if not text:
        return text
    text = clean_mentions(text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def rewrite_text(original_text: str) -> str:
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
        rewritten = clean_mentions(rewritten)
        rewritten = re.sub(r'\*\*([^*]+)\*\*', r'\1', rewritten)
        return rewritten
    except Exception as e:
        print(f"DeepSeek error: {e}")
        return original_text

async def describe_image_bytes(image_bytes: bytes) -> str:
    """
    Use GPT-4 Vision to describe the image.
    IMPORTANT: It keeps ALL betting/game details (teams, odds, scores, leagues)
    but IGNORES watermarks like @cappersfree, CF logo, and channel names.
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        if getattr(img, 'is_animated', False):
            img.seek(0)  # first frame
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{img_base64}"

        # UPDATED PROMPT – keeps details, removes watermarks
        vision_prompt = """
        Describe this image in detail for the purpose of regenerating it.

        IMPORTANT RULES:
        1. Keep all betting, sports, and game details (team names, league names, scores, odds, stake amounts, player names).
        2. Keep the overall layout, colors, style, and mood.
        3. IGNORE and DO NOT MENTION any usernames (e.g., @cappersfree), channel handles, or social media tags.
        4. IGNORE and DO NOT MENTION any logo branding (e.g., 'CF', 'Cappers Free', or any similar graphic logo).
        5. Just describe the visual scene, the text that matters (teams, odds, scores), and the composition.

        Keep the description clear and vivid, suitable as a prompt for DALL-E 3 to recreate a similar image with the same information but without the watermarks.
        """
        vision_response = openai_client.chat.completions.create(
            model="gpt-4-turbo",  # Use "gpt-4o" if this fails
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ],
            max_tokens=500
        )
        return vision_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Vision error: {e}")
        return None

async def generate_image_from_description(prompt: str) -> str:
    if not prompt or len(prompt) < 10:
        return None
    try:
        # We don't need to shorten too much – DALL-E 3 handles up to ~4000 chars
        safe_prompt = f"Create a professional betting/gaming graphic based on this description: {prompt[:500]}"
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

async def regenerate_image_from_bytes(image_bytes: bytes) -> str:
    description = await describe_image_bytes(image_bytes)
    if not description:
        return None
    return await generate_image_from_description(description)
