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

        # Detailed prompt – keep all text, numbers, team names, odds, layout
        vision_prompt = """
        Describe this image in detail. Include all visible text, numbers, team names, league names, odds, scores, stake amounts.
        Describe the layout, colors, font styles, and overall design.
        Do NOT mention any usernames (like @cappersfree) or logo branding (like CF).
        Output a clear, vivid description that would allow an image generation model to recreate this exact image without watermarks.
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
                    max_tokens=500  # increase to get more details
                )
                description = response.choices[0].message.content.strip()
                print(f"✅ Vision description: {description[:200]}...")
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

    # Use the full description – keep all details
    # Only clean up excessive whitespace, do NOT strip characters
    clean_prompt = re.sub(r'\s+', ' ', prompt).strip()
    # Limit to 500 characters to avoid token limits (DALL-E can handle more, but safe)
    if len(clean_prompt) > 500:
        clean_prompt = clean_prompt[:500] + "..."
    
    # Better instruction: recreate the described image without watermarks
    final_prompt = f"Recreate this image exactly as described, without any watermarks or logos: {clean_prompt}"

    try:
        print(f"🎨 Trying GPT Image 2 with prompt: {final_prompt[:100]}...")
        
        # CRITICAL: No 'quality' or 'response_format' parameters
        response = openai_client.images.generate(
            model="gpt-image-2",
            prompt=final_prompt,
            size="1024x1024",
            n=1
        )
        
        print(f"📦 Response received successfully")
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
        print(f"❌ OpenAI GPT Image 2 error: {e}")
        if hasattr(e, 'response'):
            print(f"📄 Response body: {e.response.text}")
        if hasattr(e, 'status_code'):
            print(f"📊 Status code: {e.status_code}")
        if hasattr(e, 'request_id'):
            print(f"🔑 Request ID: {e.request_id}")
        
        # Fallback: only try generic if the custom prompt fails with a specific error
        # But if it's a content policy violation, generic might also fail.
        # So we'll only fallback if the error is not a content policy.
        error_str = str(e).lower()
        if "content_policy" not in error_str and "moderation" not in error_str:
            try:
                print("🔄 Trying fallback with generic prompt...")
                response = openai_client.images.generate(
                    model="gpt-image-2",
                    prompt="A sports betting graphic with teams and odds, clean modern style",
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
        else:
            print("❌ Content policy violation – cannot generate.")
        
        return None

async def regenerate_image_from_bytes(image_bytes: bytes):
    description = await describe_image_bytes(image_bytes)
    if not description:
        return None
    return await generate_image_from_description(description)
