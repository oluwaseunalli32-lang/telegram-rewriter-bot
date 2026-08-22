import os
import re
import base64
import time
import asyncio
import json
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
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
        img.save(buffered, format="JPEG", quality=90)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{img_base64}"

        # EXTREME DETAIL – this is the key to accurate reconstruction
        vision_prompt = """
        You are an expert image describer. Describe the image with extreme precision as if you are creating a technical blueprint for a graphic designer.

        Follow this structure:
        1. **Overall layout**: background color (hex), size, any borders or shadows.
        2. **Text elements**: For each piece of text, list:
           - The exact text (wording, numbers, symbols)
           - Position (e.g., "top-left corner", "center", "x=20%, y=80%")
           - Font size (if discernible) and style (bold, italic, color in hex)
           - Any background box behind the text (color, border)
        3. **Graphics/Logos**: Describe any icons, lines, or shapes (ignore watermarks like @cappersfree or CF).
        4. **Colors**: Provide hex codes for all major color blocks.

        IMPORTANT: Omit any usernames (like @cappersfree) and logos (like CF) – these are watermarks you should skip.

        Output the description as plain English but make it exhaustive. The goal is to recreate the image exactly.
        """

        max_retries = 4
        base_delay = 2
        for attempt in range(max_retries):
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",  # stronger model for better accuracy
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }
                    ],
                    max_tokens=1500
                )
                description = response.choices[0].message.content.strip()
                print(f"✅ Vision description (first 300 chars): {description[:300]}...")
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
    if not prompt or len(prompt) < 20:
        return None

    now = time.time()
    elapsed = now - _last_generation_call
    if elapsed < _generation_cooldown:
        wait = _generation_cooldown - elapsed
        print(f"⏳ Waiting {wait:.1f}s before next generation...")
        await asyncio.sleep(wait)
    _last_generation_call = time.time()

    # Keep the description intact, but cap length for DALL-E (it handles ~4000 chars)
    if len(prompt) > 3500:
        prompt = prompt[:3500] + "..."

    final_prompt = f"""
    Reproduce the image exactly as described below.
    - Use the exact same layout, colors, font styles, and text positions.
    - Include all text content, numbers, team names, odds, scores.
    - Do NOT include any watermarks, usernames (like @cappersfree), or logos (like CF).
    - The background, borders, and decorative elements must match the description.

    Description:
    {prompt}
    """

    try:
        print(f"🎨 DALL-E prompt length: {len(final_prompt)} chars")
        # Log first 500 chars for debugging
        print(f"📝 DALL-E PROMPT (first 500 chars):\n{final_prompt[:500]}...")
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
                return BufferedInputFile(file=image_bytes, filename="reconstructed.png")
            elif img_data.url:
                return img_data.url
        return None
    except Exception as e:
        print(f"❌ DALL-E error: {e}")
        if hasattr(e, 'response'):
            print(f"📄 Response: {e.response.text}")
        # Fallback to a simpler prompt if content policy is triggered
        if "content_policy" in str(e).lower() or "moderation" in str(e).lower():
            try:
                print("🔄 Content policy – trying simplified prompt...")
                fallback_prompt = f"Recreate this image without watermarks: {prompt[:200]}"
                response = openai_client.images.generate(
                    model="gpt-image-2",
                    prompt=fallback_prompt,
                    size="1024x1024",
                    n=1
                )
                if response.data and response.data[0].b64_json:
                    return BufferedInputFile(file=base64.b64decode(response.data[0].b64_json), filename="fallback.png")
            except Exception as e2:
                print(f"❌ Fallback failed: {e2}")
        return None

async def regenerate_image_from_bytes(image_bytes: bytes):
    description = await describe_image_bytes(image_bytes)
    if not description:
        return None
    return await generate_image_from_description(description)
