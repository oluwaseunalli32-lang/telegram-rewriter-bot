import os
import re
import base64
import time
import asyncio
from io import BytesIO
from PIL import Image
from openai import OpenAI
from aiogram.types import BufferedInputFile

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

NEW_MENTION = os.getenv("NEW_MENTION", "")
OLD_MENTION = "@cappersfree"

_last_vision_call = 0
_vision_cooldown = 8
_last_generation_call = 0
_generation_cooldown = 8

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
                {"role": "system", "content": "Rewrite the following text to make it unique while preserving the core meaning. Return ONLY the rewritten text, nothing else. Do not add prefixes."},
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
    global _last_vision_call
    now = time.time()
    elapsed = now - _last_vision_call
    if elapsed < _vision_cooldown:
        wait = _vision_cooldown - elapsed
        print(f"⏳ Waiting {wait:.1f}s before next Vision call...")
        await asyncio.sleep(wait)
    _last_vision_call = time.time()

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
        You are a forensic image analyst. Describe the image with absolute precision.
        You MUST output a structured blueprint that an AI could use to reproduce the image exactly.

        INCLUDE:
        1. Exact text content: every word, number, team name, league, odds, score, stake, date, time.
        2. Exact position of each text element (e.g., 'top-left', 'center', 'bottom-right', 'x%, y%').
        3. Exact colors: background color, text colors, border colors (use hex codes if possible).
        4. Font styles: bold, italic, size (if discernible), font family (if known).
        5. Layout: boxes, borders, shading, gradients, rounded corners, shadows.
        6. Decorative elements: lines, icons, logos (describe them but note they are watermarks to ignore).
        7. Overall style: modern, classic, dark/light mode, etc.

        IMPORTANT: IGNORE and do NOT mention any usernames (like @cappersfree), channel names, or 'CF' logos.
        These are watermarks and should be excluded.

        Output the description in plain English, but make it so detailed that an artist could paint a perfect replica without ever seeing the original.
        """

        max_retries = 4
        base_delay = 2
        for attempt in range(max_retries):
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                description = response.choices[0].message.content.strip()
                print(f"✅ Vision description (first 300 chars): {description[:300]}...")
                # Log the full description so you can verify it's detailed
                print(f"📝 FULL VISION DESCRIPTION:\n{description}")
                return description
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    wait = base_delay * (2 ** attempt)
                    print(f"⚠️ Rate limit (429). Retry {attempt+1}/{max_retries} in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    print(f"❌ Vision error: {e}")
                    return None
        return None
    except Exception as e:
        print(f"❌ Vision preparation error: {e}")
        return None

async def generate_image_from_description(prompt: str):
    global _last_generation_call
    if not prompt or len(prompt) < 10:
        return None

    now = time.time()
    elapsed = now - _last_generation_call
    if elapsed < _generation_cooldown:
        wait = _generation_cooldown - elapsed
        print(f"⏳ Waiting {wait:.1f}s before next generation...")
        await asyncio.sleep(wait)
    _last_generation_call = time.time()

    # Keep the description intact – do NOT modify it except for length limit
    if len(prompt) > 2000:
        prompt = prompt[:2000] + "..."

    # DIRECT, NO-FLUFF PROMPT for DALL-E
    final_prompt = f"Reproduce exactly this image: {prompt}"

    try:
        print(f"🎨 DALL-E prompt length: {len(final_prompt)} chars")
        # Log the actual prompt for debugging
        print(f"📝 DALL-E PROMPT:\n{final_prompt[:500]}...")
        response = openai_client.images.generate(
            model="gpt-image-2",
            prompt=final_prompt,
            size="1024x1024",
            n=1
        )
        if response.data and len(response.data) > 0:
            img_data = response.data[0]
            if img_data.b64_json:
                image_bytes = base64.b64decode(img_data.b64_json)
                return BufferedInputFile(file=image_bytes, filename="generated.png")
            elif img_data.url:
                return img_data.url
        return None
    except Exception as e:
        print(f"❌ DALL-E error: {e}")
        if hasattr(e, 'response'):
            print(f"📄 Response: {e.response.text}")
        # Only fallback on non-content errors
        if "content_policy" not in str(e).lower() and "moderation" not in str(e).lower():
            try:
                print("🔄 Trying generic fallback...")
                resp = openai_client.images.generate(
                    model="gpt-image-2",
                    prompt="Sports betting graphic with teams and odds, clean modern layout",
                    size="1024x1024",
                    n=1
                )
                if resp.data and resp.data[0].b64_json:
                    return BufferedInputFile(file=base64.b64decode(resp.data[0].b64_json), filename="fallback.png")
            except Exception as e2:
                print(f"❌ Fallback failed: {e2}")
        return None

async def regenerate_image_from_bytes(image_bytes: bytes):
    description = await describe_image_bytes(image_bytes)
    if not description:
        return None
    return await generate_image_from_description(description)
