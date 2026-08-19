import os
import re
import base64
import time
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
                {"role": "system", "content": "Rewrite the following text to make it unique while preserving the core meaning. Return ONLY the rewritten text, nothing else. Do not add prefixes like 'Here is the rewritten text' or 'Rewritten version:'. Just output the rewritten content."},
                {"role": "user", "content": cleaned}
            ],
            temperature=0.8,
            max_tokens=2000
        )
        rewritten = response.choices[0].message.content.strip()
        for prefix in ["Here is the rewritten text:", "Rewritten version:", "Here is:", "Rewritten:"]:
            if rewritten.lower().startswith(prefix.lower()):
                rewritten = rewritten[len(prefix):].strip()
        rewritten = clean_mentions(rewritten)
        rewritten = re.sub(r'\*\*([^*]+)\*\*', r'\1', rewritten)
        return rewritten
    except Exception as e:
        print(f"DeepSeek error: {e}")
        return original_text

async def describe_image_bytes(image_bytes: bytes) -> str:
    """Use GPT-4o with exponential backoff retries for 429 errors."""
    try:
        img = Image.open(BytesIO(image_bytes))
        if getattr(img, 'is_animated', False):
            img.seek(0)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{img_base64}"

        vision_prompt = """
        Describe this image in detail for the purpose of recreating it.

        RULES:
        1. KEEP all betting, sports, and game details: team names, league names, scores, odds, stake amounts, player names, and any numbers.
        2. KEEP the overall layout, colors, style, and composition.
        3. IGNORE and DO NOT MENTION any usernames (e.g., @cappersfree), channel handles, or social media tags.
        4. IGNORE and DO NOT MENTION any logo branding (e.g., 'CF' or 'Cappers Free').
        5. Just describe the visual scene, the important text (teams, odds, scores), and the layout.
        Make the description clear and vivid, suitable as a prompt for DALL-E 3 to recreate a similar image without watermarks.
        """

        max_retries = 5
        base_delay = 1
        for attempt in range(max_retries):
            try:
                vision_response = openai_client.chat.completions.create(
                    model="gpt-4o",
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
                description = vision_response.choices[0].message.content.strip()
                print(f"Vision description (first 200 chars): {description[:200]}...")
                return description
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    wait = base_delay * (2 ** attempt)
                    print(f"Rate limited (429). Retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    print(f"Vision non‑429 error: {e}")
                    return None
        print("Vision failed after all retries (rate limit)")
        return None
    except Exception as e:
        print(f"Vision setup error: {e}")
        return None

async def generate_image_from_description(prompt: str) -> str:
    if not prompt or len(prompt) < 10:
        return None
    try:
        # Flexible: use the description directly, preserving style and details.
        final_prompt = f"Recreate the image described below, preserving its style, layout, and details, but without any watermarks or logos:\n\n{prompt}"
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=final_prompt,
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
