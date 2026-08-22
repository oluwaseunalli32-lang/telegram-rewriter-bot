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
# API CLIENTS
# ============================================================

# Kept for compatibility with your .env, but DeepSeek is NOT used
# for rewriting anymore. We need an exact username replacement,
# not an AI rewrite that can change betting information.
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ============================================================
# CONFIG
# ============================================================

NEW_MENTION = os.getenv("NEW_MENTION", "").strip()
OLD_MENTION = os.getenv("OLD_MENTION", "@cappersfree").strip()

_last_vision_call = 0.0
_vision_cooldown = 8

_last_generation_call = 0.0
_generation_cooldown = 8


# ============================================================
# TEXT PROCESSING
# ============================================================

def replace_username(text: str) -> str:
    """
    Change ONLY the old username to the new username.

    No rewriting.
    No paraphrasing.
    No removal of other @mentions.
    No changes to odds, teams, numbers, punctuation, etc.
    """
    if not text:
        return text

    if not NEW_MENTION:
        return text

    return re.sub(
        re.escape(OLD_MENTION),
        NEW_MENTION,
        text,
        flags=re.IGNORECASE,
    )


async def rewrite_text(original_text: str) -> str:
    """
    Kept with the old function name so main.py does not need to change
    its import.

    IMPORTANT:
    DeepSeek is intentionally NOT called here.

    The requirement is an exact username replacement, and deterministic
    replacement is safer than asking an LLM to rewrite the message.
    """
    return replace_username(original_text)


# ============================================================
# IMAGE VISION / DECONSTRUCTION
# ============================================================

async def describe_image_bytes(image_bytes: bytes):
    """
    Send the ORIGINAL image to OpenAI Vision and create a detailed
    reconstruction specification.

    This is the first step of the image pipeline:
        ORIGINAL IMAGE -> OPENAI VISION -> RECONSTRUCTION SPEC
    """

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

        if getattr(img, "is_animated", False):
            img.seek(0)

        if img.mode != "RGB":
            img = img.convert("RGB")

        # Keep the source reasonably sized for the vision request.
        # Do NOT destroy the important visual detail.
        max_dimension = 2048

        if max(img.size) > max_dimension:
            scale = max_dimension / max(img.size)
            new_size = (
                max(1, int(img.width * scale)),
                max(1, int(img.height * scale)),
            )
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95)

        img_base64 = base64.b64encode(
            buffered.getvalue()
        ).decode("utf-8")

        data_url = f"data:image/jpeg;base64,{img_base64}"

        vision_prompt = """
You are the visual analyst for an image recreation pipeline.

Your job is to DECONSTRUCT the supplied image so another image-generation
model can create a new image with the same overall visual structure.

IMPORTANT WATERMARK / BRANDING RULE:
- Do NOT reproduce watermarks.
- Do NOT reproduce usernames or social-media handles.
- Do NOT reproduce channel names, logos, or identifiable branding.
- Ignore those elements completely.
- If a username or watermark is present, simply state internally that it
  must be excluded from the new image. Do not put the actual username in
  the reconstruction specification.

Everything else that is legitimate content in the image should be captured.

Analyze the image carefully and provide a detailed reconstruction
specification containing:

1. CANVAS
- orientation
- approximate aspect ratio
- overall dimensions/proportions

2. EXACT CONTENT
- all visible sports/betting information
- league/competition
- team names
- player names
- matchup
- scores
- odds
- times
- dates
- stake amounts
- picks
- headings
- labels
- emojis
- all other meaningful numbers and text

Preserve the exact wording and numbers of legitimate content as closely
as possible.

3. COMPOSITION
- exact placement of major elements
- top/center/bottom sections
- left/right alignment
- margins
- spacing
- hierarchy
- cards
- panels
- dividers
- borders
- frames
- icons
- decorative elements

4. TYPOGRAPHY
- approximate font family/style
- uppercase/lowercase
- boldness
- relative sizes
- letter spacing
- alignment
- text colors
- special effects

5. COLOR PALETTE
- dominant background color
- secondary colors
- accent colors
- text colors
- gradients
- highlights

6. BACKGROUND
- texture
- grain
- stadium/sports atmosphere
- lighting
- patterns
- shadows
- glow
- depth

7. IMAGE / GRAPHIC ELEMENTS
Describe legitimate visual elements such as:
- team/player imagery
- silhouettes
- sports equipment
- stadium elements
- abstract shapes
- arrows
- badges
- icons
- lines
- effects

8. FINAL RECREATION INSTRUCTIONS
Finish with a concise but detailed set of instructions for an image
generation model explaining how to recreate the composition and style.

Do NOT shorten the analysis.
Do NOT omit details simply because they are small.
Do NOT rewrite or invent betting information.
Do NOT include watermarks, usernames, handles, logos, or channel branding
in the final reconstruction specification.
"""

        max_retries = 4
        base_delay = 2

        for attempt in range(max_retries):
            try:
                print(
                    "🔍 Sending ORIGINAL image to OpenAI Vision "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )

                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": vision_prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url,
                                        "detail": "high",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=2500,
                )

                description = (
                    response.choices[0]
                    .message.content
                    .strip()
                )

                if not description:
                    print("❌ OpenAI Vision returned an empty description.")
                    return None

                print(
                    "✅ OpenAI Vision deconstruction complete: "
                    f"{len(description)} characters"
                )
                print(
                    "🧩 Deconstruction preview: "
                    f"{description[:500]}"
                )

                return description

            except Exception as e:
                status_code = getattr(e, "status_code", None)

                if status_code == 429:
                    wait = base_delay * (2 ** attempt)

                    print(
                        f"⚠️ OpenAI Vision rate limit (429). "
                        f"Retry {attempt + 1}/{max_retries} in {wait}s"
                    )

                    await asyncio.sleep(wait)
                    continue

                print(f"❌ OpenAI Vision error: {e}")
                return None

        print("❌ OpenAI Vision failed after all retries.")
        return None

    except Exception as e:
        print(f"❌ Vision image preparation error: {e}")
        return None


# ============================================================
# IMAGE GENERATION
# ============================================================

async def generate_image_from_description(description: str):
    """
    Generate a NEW image from the complete OpenAI Vision
    reconstruction specification.

    IMPORTANT:
    The description is NOT truncated to 250 characters.
    """

    global _last_generation_call

    if not description or len(description.strip()) < 10:
        print("❌ Generation skipped: reconstruction specification is empty.")
        return None

    now = time.time()
    elapsed = now - _last_generation_call

    if elapsed < _generation_cooldown:
        wait = _generation_cooldown - elapsed

        print(
            f"⏳ Waiting {wait:.1f}s before next image generation..."
        )

        await asyncio.sleep(wait)

    _last_generation_call = time.time()

    final_prompt = f"""
Create a NEW, high-resolution sports graphic based on the reconstruction
specification below.

The supplied specification was produced by analyzing an original image.

RECREATION RULES:

- Create a NEW independently generated image.
- Match the original image's overall composition, proportions, hierarchy,
  spacing, typography style, colors, background treatment, texture,
  lighting, borders, and decorative structure as closely as possible.
- Preserve legitimate sports and betting information exactly.
- Preserve team names, player names, odds, scores, dates, times, picks,
  and other legitimate numbers/text from the specification.
- Make all important text clean, sharp, readable, and professionally
  typeset.
- Do NOT invent betting information.
- Do NOT add extra teams, odds, scores, or numbers.
- Do NOT include any watermark.
- Do NOT include any username or social-media handle.
- Do NOT include channel branding.
- Do NOT include logos or copied branding.
- If the original contained a watermark, username, handle, logo, or
  channel branding, replace that area with a visually appropriate clean
  background rather than reproducing it.
- Do not mention the reconstruction process in the generated image.

RECONSTRUCTION SPECIFICATION:

{description}
"""

    try:
        print(
            "🎨 Generating NEW image from FULL reconstruction specification..."
        )
        print(
            f"📝 Generation prompt length: {len(final_prompt)} characters"
        )

        response = openai_client.images.generate(
            model="gpt-image-2",
            prompt=final_prompt,
            n=1,
        )

        if not response.data:
            print("❌ OpenAI image generation returned no data.")
            return None

        img_data = response.data[0]

        if getattr(img_data, "b64_json", None):
            image_bytes = base64.b64decode(
                img_data.b64_json
            )

            print(
                f"✅ New image generated: {len(image_bytes)} bytes"
            )

            return BufferedInputFile(
                file=image_bytes,
                filename="regenerated.png",
            )

        if getattr(img_data, "url", None):
            print("✅ New image generated as URL.")
            return img_data.url

        print("❌ Image response contained neither b64_json nor URL.")
        return None

    except Exception as e:
        print(f"❌ OpenAI image generation error: {e}")

        if hasattr(e, "response"):
            try:
                print(f"📄 API response: {e.response.text}")
            except Exception:
                pass

        return None


# ============================================================
# FULL IMAGE PIPELINE
# ============================================================

async def regenerate_image_from_bytes(image_bytes: bytes):
    """
    Complete image pipeline:

        Telegram image
            ↓
        OpenAI Vision
            ↓
        Detailed deconstruction
            ↓
        OpenAI image generation
            ↓
        New clean image
    """

    if not image_bytes:
        print("❌ No image bytes supplied.")
        return None

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🖼️ IMAGE REGENERATION PIPELINE START")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # STEP 1: Deconstruct the ORIGINAL image.
    print("1️⃣ STEP 1/2 — OpenAI Vision deconstruction...")

    description = await describe_image_bytes(image_bytes)

    if not description:
        print("❌ STEP 1 FAILED — no reconstruction specification.")
        return None

    # STEP 2: Generate a NEW image from that specification.
    print("2️⃣ STEP 2/2 — OpenAI image generation...")

    generated = await generate_image_from_description(
        description
    )

    if generated:
        print("✅ IMAGE REGENERATION PIPELINE COMPLETE")
    else:
        print("❌ STEP 2 FAILED — image generation failed.")

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return generated
