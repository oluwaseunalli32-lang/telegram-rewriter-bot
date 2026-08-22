import os
import re
import base64
import time
import asyncio
from io import BytesIO

from PIL import Image
from openai import OpenAI
from aiogram.types import BufferedInputFile


# ============================================================
# CLIENTS
# ============================================================

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ============================================================
# CONFIG
# ============================================================

NEW_MENTION = os.getenv("NEW_MENTION", "").strip()
OLD_MENTION = os.getenv("OLD_MENTION", "@cappersfree").strip()

# OpenAI models
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5.6-luna")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-2")

_last_vision_call = 0.0
_vision_cooldown = 8

_last_generation_call = 0.0
_generation_cooldown = 8


# ============================================================
# USERNAME REPLACEMENT
# DeepSeek's ONLY job here is replacing OLD_MENTION with
# NEW_MENTION. A deterministic validation/fallback guarantees
# that no other text is changed.
# ============================================================

def replace_username_locally(text: str) -> str:
    if not text or not OLD_MENTION or not NEW_MENTION:
        return text

    return re.sub(
        re.escape(OLD_MENTION),
        NEW_MENTION,
        text,
        flags=re.IGNORECASE,
    )


def same_except_username(original: str, candidate: str) -> bool:
    """
    Returns True only when replacing OLD_MENTION with NEW_MENTION
    in the original produces exactly the candidate.
    """
    expected = replace_username_locally(original)
    return candidate == expected


async def rewrite_text(original_text: str) -> str:
    """
    DeepSeek is NOT allowed to rewrite, paraphrase, summarize,
    reorder, or modify anything except the configured username.
    """
    if not original_text:
        return original_text

    # Nothing to replace -> don't waste an API call.
    if (
        not OLD_MENTION
        or not NEW_MENTION
        or not re.search(re.escape(OLD_MENTION), original_text, flags=re.IGNORECASE)
    ):
        return original_text

    system_prompt = (
        "You are an exact text replacement engine. "
        "Do NOT rewrite, paraphrase, correct grammar, summarize, "
        "format, add, remove, reorder, or change any text. "
        f"Replace every case-insensitive occurrence of {OLD_MENTION!r} "
        f"with {NEW_MENTION!r}. "
        "Every other character must remain exactly identical. "
        "Return ONLY the resulting text."
    )

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_text},
            ],
            temperature=0,
            max_tokens=max(100, min(4000, len(original_text) * 2)),
        )

        result = (response.choices[0].message.content or "").strip()

        # DeepSeek must not alter anything else.
        if same_except_username(original_text, result):
            return result

        print("⚠️ DeepSeek changed more than the username. Using exact local replacement.")
        return replace_username_locally(original_text)

    except Exception as e:
        print(f"❌ DeepSeek username replacement error: {e}")
        # Guaranteed exact fallback.
        return replace_username_locally(original_text)


# ============================================================
# IMAGE PREPARATION
# ============================================================

def image_bytes_to_data_url(image_bytes: bytes) -> str:
    img = Image.open(BytesIO(image_bytes))

    if getattr(img, "is_animated", False):
        img.seek(0)

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    # Flatten transparency onto white before JPEG conversion.
    if img.mode == "RGBA":
        background = Image.new("RGB", img.size, "white")
        background.paste(img, mask=img.getchannel("A"))
        img = background

    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=92, optimize=True)

    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# ============================================================
# OPENAI STEP 1:
# DECONSTRUCT / EXPLAIN THE SOURCE IMAGE
#
# The model extracts reusable visual information. It must ignore
# usernames, handles, logos, signatures, and watermark elements.
# ============================================================

async def describe_image_bytes(image_bytes: bytes) -> str | None:
    global _last_vision_call

    elapsed = time.time() - _last_vision_call
    if elapsed < _vision_cooldown:
        wait = _vision_cooldown - elapsed
        print(f"⏳ Waiting {wait:.1f}s before next Vision call...")
        await asyncio.sleep(wait)

    _last_vision_call = time.time()

    try:
        data_url = image_bytes_to_data_url(image_bytes)

        vision_prompt = """
Analyze this image as a reference for creating a NEW, original but visually similar graphic.

First deconstruct the image in detail. Describe:
1. Canvas orientation and approximate aspect ratio.
2. Background colors, texture, lighting, gradients, patterns, and atmosphere.
3. Layout: position and hierarchy of all important visual sections.
4. Typography: approximate font personality, weight, capitalization, alignment, spacing, and size hierarchy.
5. All visible content that is important to the graphic: titles, team names, league names, odds, scores, dates, times, player names, stakes, and other numbers.
6. Shapes, borders, cards, panels, dividers, icons, arrows, glow, shadows, and accent colors.
7. The overall design style and visual mood.

IMPORTANT:
- Do NOT include or reproduce any watermark.
- Ignore usernames, @handles, channel names, social media tags, logos, signatures, and branding marks.
- Do not treat those ignored elements as part of the new design.
- Preserve only legitimate content and reusable visual/layout characteristics.
- This is analysis for generating a NEW image, not instructions to edit the source image.

Return a detailed, structured design brief that can be given directly to an image generation model.
""".strip()

        max_retries = 4
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = openai_client.responses.create(
                    model=VISION_MODEL,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": vision_prompt,
                                },
                                {
                                    "type": "input_image",
                                    "image_url": data_url,
                                    "detail": "high",
                                },
                            ],
                        }
                    ],
                )

                description = (response.output_text or "").strip()

                if description:
                    print(f"✅ Image design brief created ({len(description)} chars)")
                    return description

                print("❌ Vision returned an empty design brief.")
                return None

            except Exception as e:
                status = getattr(e, "status_code", None)

                if status == 429 and attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt)
                    print(
                        f"⚠️ Rate limit. Retry {attempt + 1}/{max_retries} "
                        f"in {wait}s..."
                    )
                    await asyncio.sleep(wait)
                    continue

                print(f"❌ OpenAI vision error: {e}")
                return None

        return None

    except Exception as e:
        print(f"❌ Image preparation error: {e}")
        return None


# ============================================================
# OPENAI STEP 2:
# GENERATE A NEW SIMILAR IMAGE FROM THE DESIGN BRIEF
#
# This intentionally creates a fresh image from the extracted
# description rather than editing the original source image.
# ============================================================

async def generate_image_from_description(description: str):
    global _last_generation_call

    if not description or len(description.strip()) < 10:
        return None

    elapsed = time.time() - _last_generation_call
    if elapsed < _generation_cooldown:
        wait = _generation_cooldown - elapsed
        print(f"⏳ Waiting {wait:.1f}s before image generation...")
        await asyncio.sleep(wait)

    _last_generation_call = time.time()

    generation_prompt = f"""
Create a NEW, original high-quality sports betting/social media graphic using the following design brief as visual guidance.

{description}

Requirements:
- Recreate the overall visual concept, information hierarchy, layout logic, color palette, typography hierarchy, texture, and mood described in the brief.
- Generate a fresh composition rather than copying pixels from the reference.
- Keep all legitimate betting/game information from the design brief accurate.
- Do NOT include any watermark, @username, social handle, channel name, logo, signature, or branding mark that was present in the reference.
- Do NOT add a replacement watermark or invented branding.
- Keep text crisp, readable, correctly spelled, and well aligned.
- Do not add extra betting picks, teams, odds, scores, or numbers that were not present in the design brief.
- Output only the finished graphic.
""".strip()

    try:
        print(f"🎨 Generating with {IMAGE_MODEL}...")
        print(f"📝 Generation prompt length: {len(generation_prompt)} chars")

        response = openai_client.images.generate(
            model=IMAGE_MODEL,
            prompt=generation_prompt,
            n=1,
        )

        if not response.data:
            print("❌ Image generation returned no data.")
            return None

        image_data = response.data[0]

        if getattr(image_data, "b64_json", None):
            generated_bytes = base64.b64decode(image_data.b64_json)
            return BufferedInputFile(
                file=generated_bytes,
                filename="generated.png",
            )

        if getattr(image_data, "url", None):
            return image_data.url

        print("❌ Image generation returned neither b64_json nor url.")
        return None

    except Exception as e:
        print(f"❌ OpenAI image generation error: {e}")

        response_obj = getattr(e, "response", None)
        if response_obj is not None:
            try:
                print(f"📄 Response: {response_obj.text}")
            except Exception:
                pass

        return None


# ============================================================
# COMPLETE PIPELINE
# ============================================================

async def regenerate_image_from_bytes(image_bytes: bytes):
    """
    1. OpenAI analyzes and deconstructs the reference image.
    2. OpenAI produces a detailed reusable design brief.
    3. GPT Image generates a NEW similar graphic from that brief.
    """
    description = await describe_image_bytes(image_bytes)

    if not description:
        return None

    print("📋 Design brief ready. Starting new image generation...")
    return await generate_image_from_description(description)
