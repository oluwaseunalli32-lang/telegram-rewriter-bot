import os
import re
import base64
import time
import asyncio
from io import BytesIO
from PIL import Image
from openai import OpenAI
from aiogram.types import BufferedInputFile

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

# Rate-limit protection
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
        Describe this image concisely. Keep all sports/betting details: team names, leagues, odds, scores, stakes.
        Describe the layout, colors, and style.
        Ignore any usernames (like @cappersfree) and logos (like CF).
        Output a short description suitable for an image generation model to recreate the image without watermarks.
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

async def generate_image_from_description(prompt: str):
    """Generate image using gpt-image-2 with proper error logging."""
    global _last_generation_call
    if not prompt or len(prompt) < 10:
        print("⚠️ Prompt too short for image generation.")
        return None

    now = time.time()
    elapsed = now - _last_generation_call
    if elapsed < _generation_cooldown:
        wait = _generation_cooldown - elapsed
        print(f"⏳ Waiting {wait:.1f}s before next generation...")
        await asyncio.sleep(wait)
    _last_generation_call = time.time()

    # Clean and shorten prompt – gpt-image-2 is strict
    clean_prompt = re.sub(r'[^a-zA-Z0-9\s\-+]', '', prompt)
    if len(clean_prompt) > 200:
        clean_prompt = clean_prompt[:200] + "..."
    final_prompt = f"Sports betting graphic: {clean_prompt}"

    try:
        print(f"🎨 Trying GPT Image 2 with prompt: {final_prompt[:80]}...")
        
        # CRITICAL: Do NOT include 'quality' or 'response_format' parameters
        # gpt-image-2 rejects them with 400 error
        response = openai_client.images.generate(
            model="gpt-image-2",
            prompt=final_prompt,
            size="1024x1024",
            n=1
        )
        
        # Log the full response for debugging
        print(f"📦 Response status: success")
        if hasattr(response, '_request_id'):
            print(f"🔑 Request ID: {response._request_id}")

        if response.data and len(response.data) > 0:
            img_data = response.data[0]
            if img_data.b64_json:
                image_bytes = base64.b64decode(img_data.b64_json)
                input_file = BufferedInputFile(file=image_bytes, filename="generated.png")
                print("✅ Generated image from base64")
                return input_file
            elif img_data.url:
                print(f"✅ Generated image URL: {img_data.url}")
                return img_data.url
        print("❌ No image data in response.")
        return None
        
    except Exception as e:
        # Print FULL error details for debugging
        print(f"❌ OpenAI GPT Image 2 error: {e}")
        
        # Try to extract more details from the error
        if hasattr(e, 'response'):
            print(f"📄 Response body: {e.response.text}")
        if hasattr(e, 'status_code'):
            print(f"📊 Status code: {e.status_code}")
        if hasattr(e, 'request_id'):
            print(f"🔑 Request ID: {e.request_id}")
        
        # Fallback: try with an even simpler prompt
        try:
            print("🔄 Trying fallback with generic prompt...")
            response = openai_client.images.generate(
                model="gpt-image-2",
                prompt="Sports betting odds graphic, clean modern style",
                size="1024x1024",
                n=1
            )
            if response.data and len(response.data) > 0:
                img_data = response.data[0]
                if img_data.b64_json:
                    image_bytes = base64.b64decode(img_data.b64_json)
                    print("✅ Fallback succeeded")
                    return BufferedInputFile(file=image_bytes, filename="fallback.png")
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            if hasattr(e2, 'response'):
                print(f"📄 Fallback response: {e2.response.text}")
        
        return None

async def regenerate_image_from_bytes(image_bytes: bytes):
    description = await describe_image_bytes(image_bytes)
    if not description:
        return None
    return await generate_image_from_description(description)
