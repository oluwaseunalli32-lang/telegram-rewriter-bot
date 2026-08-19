import os
import re
import base64
import time
import asyncio
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

# Rate limit protection
_last_vision_call = 0
_vision_cooldown = 8  # seconds between Vision calls
_last_dalle_call = 0
_dalle_cooldown = 10  # seconds between DALL-E calls

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
    """Use gpt-4o-mini for Vision with rate limiting and retries."""
    global _last_vision_call

    # Enforce cooldown
    now = time.time()
    elapsed = now - _last_vision_call
    if elapsed < _vision_cooldown:
        wait = _vision_cooldown - elapsed
        print(f"⏳ Waiting {wait:.1f}s before next Vision call...")
        await asyncio.sleep(wait)
    _last_vision_call = time.time()

    try:
        # Prepare image
        img = Image.open(BytesIO(image_bytes))
        if getattr(img, 'is_animated', False):
            img.seek(0)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{img_base64}"

        # Short, clear prompt
        vision_prompt = """
        Describe this image concisely. Keep all sports/betting details: team names, leagues, odds, scores, stakes.
        Describe the layout, colors, and style.
        Ignore any usernames (like @cappersfree) and logos (like CF).
        Output a short description suitable for DALL-E to recreate the image without watermarks.
        """

        max_retries = 4
        base_delay = 2
        for attempt in range(max_retries):
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # faster, cheaper, good enough for this task
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }
                    ],
                    max_tokens=300
                )
                description = response.choices[0].message.content.strip()
                print(f"✅ Vision description: {description[:150]}...")
                return description
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    wait = base_delay * (2 ** attempt)
                    print(f"⚠️ Rate limit (429). Retry {attempt+1}/{max_retries} in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    print(f"❌ Vision error: {e}")
                    return None
        print("❌ Vision failed after all retries")
        return None
    except Exception as e:
        print(f"❌ Vision preparation error: {e}")
        return None

async def generate_image_from_description(prompt: str) -> str:
    """Generate image with DALL-E 3, respecting cooldown."""
    global _last_dalle_call

    if not prompt or len(prompt) < 10:
        return None

    # Enforce cooldown
    now = time.time()
    elapsed = now - _last_dalle_call
    if elapsed < _dalle_cooldown:
        wait = _dalle_cooldown - elapsed
        print(f"⏳ Waiting {wait:.1f}s before next DALL-E call...")
        await asyncio.sleep(wait)
    _last_dalle_call = time.time()

    try:
        # Clean prompt: remove non-ASCII, limit length
        clean_prompt = re.sub(r'[^\x00-\x7F]+', '', prompt)  # remove emojis/special chars
        if len(clean_prompt) > 300:
            clean_prompt = clean_prompt[:300] + "..."
        final_prompt = f"Recreate this image without any watermarks or logos: {clean_prompt}"
        print(f"🎨 DALL-E prompt: {final_prompt[:100]}...")
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=final_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        print(f"❌ DALL-E error: {e}")
        return None

async def regenerate_image_from_bytes(image_bytes: bytes) -> str:
    description = await describe_image_bytes(image_bytes)
    if not description:
        return None
    return await generate_image_from_description(description)
