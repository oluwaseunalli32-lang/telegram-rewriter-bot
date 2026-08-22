import os
import re
import base64
import asyncio
import logging
from io import BytesIO
from typing import Optional, Union

from PIL import Image
from openai import OpenAI
from aiogram.types import BufferedInputFile


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# OPENAI CONFIGURATION
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "gpt-4o-mini",
).strip()

IMAGE_MODEL = os.getenv(
    "IMAGE_MODEL",
    "gpt-image-2",
).strip()

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Add it to Render Environment Variables."
    )

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=180.0,
    max_retries=0,
)


# ============================================================
# CAPTION CONFIGURATION
# ============================================================

OLD_MENTION = os.getenv(
    "OLD_MENTION",
    "@cappersfree",
).strip()

NEW_MENTION = os.getenv(
    "NEW_MENTION",
    "",
).strip()

if NEW_MENTION and not NEW_MENTION.startswith("@"):
    NEW_MENTION = "@" + NEW_MENTION


# ============================================================
# STARTUP LOG
# ============================================================

logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
logger.info("🤖 AI PROCESSOR LOADED")
logger.info("👤 OLD_MENTION = %r", OLD_MENTION)
logger.info("👤 NEW_MENTION = %r", NEW_MENTION)
logger.info("🧠 VISION_MODEL = %s", VISION_MODEL)
logger.info("🎨 IMAGE_MODEL = %s", IMAGE_MODEL)
logger.info("✏️ Caption AI rewriting = DISABLED")
logger.info("🖼️ Image pipeline = VISION → DECONSTRUCTION → GENERATION")
logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ============================================================
# CAPTION PROCESSING
# ============================================================

def replace_username(text: str) -> str:
    """
    Caption processing is deliberately NOT AI-powered.

    Rules:
      1. Remove every literal '*' character.
      2. Replace @cappersfree with NEW_MENTION.
      3. Leave everything else unchanged.
    """

    if not text:
        return text

    result = text.replace("*", "")

    if not NEW_MENTION:
        logger.warning(
            "⚠️ NEW_MENTION is empty; username replacement skipped."
        )
        return result

    old_username = OLD_MENTION.lstrip("@").strip()

    if not old_username:
        return result

    # After '*' removal, replace the exact old username.
    # Case-insensitive so @CappersFree is also handled.
    pattern = r"@" + re.escape(old_username)

    result = re.sub(
        pattern,
        NEW_MENTION,
        result,
        flags=re.IGNORECASE,
    )

    return result


async def rewrite_text(original_text: str) -> str:
    """
    Kept with the same function name expected by main.py.

    No DeepSeek.
    No OpenAI.
    No paraphrasing.
    """

    result = replace_username(original_text)

    logger.info("📝 Caption original: %r", original_text)
    logger.info("📝 Caption final:    %r", result)

    return result


# ============================================================
# IMAGE PREPARATION
# ============================================================

def resize_for_vision(image: Image.Image) -> Image.Image:
    """Keep Vision uploads reasonably sized."""

    max_dimension = 2048

    if max(image.size) <= max_dimension:
        return image

    scale = max_dimension / max(image.size)

    new_size = (
        max(1, int(image.width * scale)),
        max(1, int(image.height * scale)),
    )

    return image.resize(
        new_size,
        Image.Resampling.LANCZOS,
    )


def flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Flatten transparency onto white."""

    rgba = image.convert("RGBA")

    background = Image.new(
        "RGB",
        rgba.size,
        "white",
    )

    background.paste(
        rgba,
        mask=rgba.getchannel("A"),
    )

    return background


def prepare_image_for_vision(
    image_bytes: bytes,
) -> Optional[Image.Image]:
    """
    Convert a normal image or GIF into a Vision-friendly image.

    Static image:
        image -> JPEG

    Animated GIF:
        representative frames -> contact sheet -> JPEG
    """

    try:
        source = Image.open(BytesIO(image_bytes))

        is_animated = bool(
            getattr(source, "is_animated", False)
        )

        # ----------------------------------------------------
        # STATIC IMAGE
        # ----------------------------------------------------

        if not is_animated:
            image = flatten_to_rgb(source)
            return resize_for_vision(image)

        # ----------------------------------------------------
        # ANIMATED GIF
        # ----------------------------------------------------

        total_frames = int(
            getattr(source, "n_frames", 1)
        )

        sample_count = min(
            6,
            total_frames,
        )

        if sample_count <= 1:
            frame_indexes = [0]
        else:
            frame_indexes = [
                round(
                    i * (total_frames - 1)
                    / (sample_count - 1)
                )
                for i in range(sample_count)
            ]

        frames = []

        for frame_number in frame_indexes:
            try:
                source.seek(frame_number)

                frame = flatten_to_rgb(source)
                frame = resize_for_vision(frame)

                frames.append(frame.copy())

            except Exception:
                logger.exception(
                    "⚠️ Could not read GIF frame %s",
                    frame_number,
                )

        if not frames:
            logger.error(
                "❌ GIF contained no readable frames."
            )
            return None

        logger.info(
            "🎞️ GIF detected: %s total frames; using %s representative frames.",
            total_frames,
            len(frames),
        )

        # ----------------------------------------------------
        # CONTACT SHEET
        # ----------------------------------------------------

        columns = min(3, len(frames))

        rows = (
            len(frames) + columns - 1
        ) // columns

        cell_width = max(
            frame.width for frame in frames
        )

        cell_height = max(
            frame.height for frame in frames
        )

        padding = 20

        sheet_width = (
            columns * cell_width
            + (columns + 1) * padding
        )

        sheet_height = (
            rows * cell_height
            + (rows + 1) * padding
        )

        sheet = Image.new(
            "RGB",
            (sheet_width, sheet_height),
            "white",
        )

        for index, frame in enumerate(frames):
            x = (
                padding
                + (index % columns)
                * (cell_width + padding)
            )

            y = (
                padding
                + (index // columns)
                * (cell_height + padding)
            )

            sheet.paste(
                frame,
                (x, y),
            )

        return resize_for_vision(sheet)

    except Exception:
        logger.exception(
            "❌ IMAGE PREPARATION FAILED"
        )
        return None


def image_to_data_url(
    image: Image.Image,
) -> str:
    """Convert prepared image to a base64 JPEG data URL."""

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# OPENAI VISION
# ============================================================

VISION_PROMPT = """
Analyze this image carefully and create a detailed
reconstruction specification for a NEW image.

The goal is to understand the visual design and legitimate
content so an image-generation model can create a new,
similar graphic.

IMPORTANT:
- Do not reproduce watermarks.
- Do not reproduce usernames or @handles.
- Do not reproduce channel branding.
- Do not reproduce third-party logos or brand marks.
- Describe those areas generically.
- Do not invent information that is not visible.

Describe:

1. CANVAS
- orientation
- approximate aspect ratio
- dimensions/proportions
- overall layout

2. MAIN CONTENT
- people
- athletes
- teams
- sports
- objects
- equipment
- backgrounds
- locations
- cards
- panels
- icons
- graphics

3. LEGITIMATE TEXT
Transcribe visible legitimate text accurately:
- headings
- matchups
- scores
- odds
- picks
- dates
- times
- labels
- numbers

Do NOT include usernames, @handles, watermarks, channel names,
or branding.

4. COMPOSITION
- top, center, bottom sections
- left/right placement
- alignment
- spacing
- margins
- cards
- panels
- borders
- dividers

5. TYPOGRAPHY
- font style
- weight
- size hierarchy
- case
- alignment
- colors
- outlines
- shadows
- glow
- spacing

6. COLORS
- background
- primary colors
- secondary colors
- accents
- gradients
- highlights
- shadows

7. BACKGROUND
- texture
- lighting
- atmosphere
- depth
- blur
- patterns
- gradients
- environmental details

8. GRAPHIC ELEMENTS
- icons
- arrows
- shapes
- lines
- badges
- panels
- borders
- decorative elements

9. ANIMATION
If this is a contact sheet made from a GIF:
- describe what changes between frames
- describe movement
- describe transitions
- identify fixed vs changing elements
- describe the overall animation concept

10. FINAL GENERATION SPECIFICATION
Finish with a concise but detailed specification another
image generator can use to create a NEW image with similar:
- composition
- hierarchy
- visual style
- color relationships
- legitimate information
- design language

while excluding watermarks, usernames, @handles, channel
branding, and third-party logos.

Do not invent information.
"""


async def describe_image_bytes(
    image_bytes: bytes,
) -> Optional[str]:
    """
    Analyze the original image/GIF with OpenAI Vision.
    """

    if not image_bytes:
        logger.error(
            "❌ Vision received empty image bytes."
        )
        return None

    logger.info(
        "🖼️ Preparing original media for Vision..."
    )

    try:
        prepared = prepare_image_for_vision(
            image_bytes
        )

        if prepared is None:
            logger.error(
                "❌ Vision image preparation returned None."
            )
            return None

        image_url = image_to_data_url(
            prepared
        )

        logger.info(
            "🔍 Sending original media to OpenAI Vision..."
        )

        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": VISION_PROMPT,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4000,
        )

        description = (
            response.choices[0]
            .message.content
        )

        if not description:
            logger.error(
                "❌ Vision returned empty content."
            )
            return None

        description = description.strip()

        logger.info(
            "✅ Vision deconstruction complete (%d chars).",
            len(description),
        )

        logger.debug(
            "🧩 Vision specification preview: %s",
            description[:1500],
        )

        return description

    except Exception as e:
        logger.exception(
            "❌ OPENAI VISION FAILED | type=%s | status=%s | error=%s",
            type(e).__name__,
            getattr(e, "status_code", None),
            str(e),
        )

        body = getattr(e, "body", None)

        if body:
            logger.error(
                "❌ OpenAI Vision error body: %r",
                body,
            )

        return None


# ============================================================
# IMAGE GENERATION
# ============================================================

def build_generation_prompt(
    description: str,
) -> str:
    return f"""
Create a NEW professional sports graphic using the
reconstruction specification below.

This must be a newly generated image, not a direct copy.

PRESERVE:
- overall composition
- visual hierarchy
- spacing
- visual structure
- color relationships
- typography style
- background treatment
- legitimate information

TEXT:
Make legitimate visible text as accurate and readable as
possible. Do not invent scores, odds, dates, teams, players,
picks, or numbers.

DO NOT INCLUDE:
- watermarks
- usernames
- @handles
- social-media handles
- channel branding
- third-party logos
- third-party brand marks

Where such material existed, replace it with a clean,
natural part of the new design.

QUALITY:
- sharp
- professional
- clean typography
- clean spacing
- strong hierarchy
- polished sports-graphic appearance
- coherent colors
- high visual quality

RECONSTRUCTION SPECIFICATION:
{description}
"""


async def generate_image_from_description(
    description: str,
) -> Optional[BufferedInputFile]:
    """
    Generate a new image from the Vision specification.
    """

    if not description:
        logger.error(
            "❌ Cannot generate image: description is empty."
        )
        return None

    prompt = build_generation_prompt(
        description
    )

    logger.info(
        "🎨 Sending reconstruction specification to image generation..."
    )

    logger.info(
        "📝 Generation prompt length: %d characters",
        len(prompt),
    )

    try:
        response = await asyncio.to_thread(
            openai_client.images.generate,
            model=IMAGE_MODEL,
            prompt=prompt,
            n=1,
        )

        if not response:
            logger.error(
                "❌ OpenAI image generation returned no response."
            )
            return None

        if not response.data:
            logger.error(
                "❌ OpenAI image generation returned no data."
            )
            return None

        item = response.data[0]

        b64_json = getattr(
            item,
            "b64_json",
            None,
        )

        if not b64_json:
            logger.error(
                "❌ Image response did not contain b64_json."
            )
            logger.error(
                "❌ Response item type: %s",
                type(item).__name__,
            )
            return None

        try:
            image_bytes = base64.b64decode(
                b64_json,
                validate=True,
            )
        except Exception:
            logger.exception(
                "❌ Failed to decode generated image."
            )
            return None

        if not image_bytes:
            logger.error(
                "❌ Generated image bytes are empty."
            )
            return None

        logger.info(
            "✅ NEW IMAGE GENERATED (%d bytes)",
            len(image_bytes),
        )

        return BufferedInputFile(
            file=image_bytes,
            filename="regenerated.png",
        )

    except Exception as e:
        logger.exception(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        logger.exception(
            "❌ OPENAI IMAGE GENERATION FAILED"
        )
        logger.exception(
            "❌ Type: %s",
            type(e).__name__,
        )
        logger.exception(
            "❌ Status: %s",
            getattr(e, "status_code", None),
        )
        logger.exception(
            "❌ Error: %s",
            str(e),
        )

        body = getattr(e, "body", None)

        if body:
            logger.error(
                "❌ API ERROR BODY: %r",
                body,
            )

        code = getattr(e, "code", None)

        if code:
            logger.error(
                "❌ API ERROR CODE: %r",
                code,
            )

        param = getattr(e, "param", None)

        if param:
            logger.error(
                "❌ API ERROR PARAM: %r",
                param,
            )

        logger.exception(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        return None


# ============================================================
# COMPLETE REGENERATION PIPELINE
# ============================================================

async def regenerate_image_from_bytes(
    image_bytes: bytes,
) -> Optional[BufferedInputFile]:
    """
    Complete pipeline:

        Telegram image/GIF
                ↓
        OpenAI Vision
                ↓
        Deconstruction
                ↓
        OpenAI image generation
                ↓
        NEW IMAGE

    The original image is never returned as a fallback.
    """

    if not image_bytes:
        logger.error(
            "❌ regenerate_image_from_bytes received no bytes."
        )
        return None

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    logger.info(
        "🖼️ IMAGE REGENERATION PIPELINE START"
    )
    logger.info(
        "1️⃣ STEP 1/2 — OPENAI VISION → DECONSTRUCTION"
    )

    description = await describe_image_bytes(
        image_bytes
    )

    if not description:
        logger.error(
            "❌ STEP 1 FAILED — Vision/deconstruction returned no specification."
        )
        return None

    logger.info(
        "✅ STEP 1 COMPLETE — deconstruction received."
    )

    logger.info(
        "2️⃣ STEP 2/2 — OPENAI IMAGE GENERATION"
    )

    generated = await generate_image_from_description(
        description
    )

    if not generated:
        logger.error(
            "❌ STEP 2 FAILED — image generation returned no image."
        )
        return None

    logger.info(
        "✅ STEP 2 COMPLETE — NEW IMAGE READY."
    )
    logger.info(
        "✅ IMAGE REGENERATION PIPELINE COMPLETE"
    )
    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return generated
