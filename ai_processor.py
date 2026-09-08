import os
import re
import io
import base64
import logging

from PIL import Image, ImageChops, ImageFilter
from openai import AsyncOpenAI

logger = logging.getLogger("ai_processor")

# ============================================================
# CONFIG
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

OPENAI_IMAGE_MODEL = os.getenv(
    "OPENAI_IMAGE_MODEL",
    "gpt-image-2"
).strip()

OPENAI_IMAGE_QUALITY = os.getenv(
    "OPENAI_IMAGE_QUALITY",
    "low"
).strip()

NEW_MENTION = os.getenv(
    "NEW_MENTION",
    "@PrimeAnalysiss"
).strip()

if NEW_MENTION and not NEW_MENTION.startswith("@"):
    NEW_MENTION = "@" + NEW_MENTION

OLD_MENTION = "@cappersfree"

client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================
# CAPTION HANDLING
# ============================================================

def replace_username(text: str | None) -> str | None:
    """
    Caption rules:
    1. Remove *
    2. Replace @cappersfree with NEW_MENTION
    3. Change nothing else
    """
    if not text:
        return text

    result = text.replace("*", "")

    result = re.sub(
        re.escape(OLD_MENTION),
        NEW_MENTION,
        result,
        flags=re.IGNORECASE,
    )

    return result


async def rewrite_text(original_text: str | None) -> str | None:
    """
    Kept for compatibility with main.py.
    No AI is used for captions.
    """
    return replace_username(original_text)


# ============================================================
# LOCAL WATERMARK DETECTION
# ============================================================

def _create_watermark_mask(image: Image.Image) -> Image.Image:
    """
    Creates a rough mask around common red/faded-red watermark pixels.

    This is intentionally done locally so we DON'T spend an OpenAI
    image-edit request on images that don't appear to contain the
    watermark.
    """

    rgb = image.convert("RGB")

    # Work at reduced resolution for cheaper/faster local detection.
    max_side = 1200

    scale = min(
        1.0,
        max_side / max(rgb.size)
    )

    if scale < 1.0:
        small = rgb.resize(
            (
                max(1, int(rgb.width * scale)),
                max(1, int(rgb.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    else:
        small = rgb

    pixels = small.load()

    mask = Image.new(
        "L",
        small.size,
        0,
    )

    mask_pixels = mask.load()

    for y in range(small.height):
        for x in range(small.width):
            r, g, b = pixels[x, y]

            # Strong red / faded red detection.
            red_score = r - max(g, b)

            is_red = (
                r >= 115
                and red_score >= 25
                and r >= g * 1.12
                and r >= b * 1.12
            )

            # Dark red / greyish red variants.
            dark_red = (
                r >= 75
                and red_score >= 18
                and r >= g * 1.08
                and r >= b * 1.08
            )

            if is_red or dark_red:
                mask_pixels[x, y] = 255

    # Join broken letters / logo fragments.
    mask = mask.filter(
        ImageFilter.MaxFilter(5)
    )

    mask = mask.filter(
        ImageFilter.GaussianBlur(1.2)
    )

    # Return at original image size.
    if scale < 1.0:
        mask = mask.resize(
            rgb.size,
            Image.Resampling.BICUBIC,
        )

    return mask


def _mask_has_enough_pixels(mask: Image.Image) -> bool:
    """
    Avoid sending images to OpenAI where almost nothing was detected.
    """

    histogram = mask.histogram()

    white_pixels = sum(histogram[180:256])
    total_pixels = mask.width * mask.height

    if total_pixels <= 0:
        return False

    percentage = white_pixels / total_pixels

    logger.info(
        "🔎 Possible watermark coverage: %.4f%%",
        percentage * 100,
    )

    # Very small = probably no watermark.
    # Very large = likely not a watermark and should not be edited.
    return (
        percentage >= 0.00005
        and percentage <= 0.08
    )


# ============================================================
# IMAGE PREPARATION
# ============================================================

def _prepare_image(image_bytes: bytes):
    """
    Converts incoming Telegram image into a safe PNG/JPEG-compatible
    format for the image editing API.
    """

    source = Image.open(io.BytesIO(image_bytes))

    # Flatten animated formats to first frame.
    if getattr(source, "is_animated", False):
        source.seek(0)

    source = source.convert("RGB")

    # Keep requests reasonably sized.
    max_dimension = 1536

    if max(source.size) > max_dimension:
        ratio = max_dimension / max(source.size)

        new_size = (
            max(1, int(source.width * ratio)),
            max(1, int(source.height * ratio)),
        )

        source = source.resize(
            new_size,
            Image.Resampling.LANCZOS,
        )

    return source


# ============================================================
# OPENAI IMAGE EDIT
# ============================================================

async def remove_watermarks_from_bytes(
    image_bytes: bytes,
    filename: str = "image.jpg",
):
    """
    Remove the @cappersfree / CF watermark using a single
    OpenAI image-edit request.

    Returns:
        bytes
        OR None when the image should remain unchanged.
    """

    if not image_bytes:
        return None

    if client is None:
        logger.error(
            "❌ OPENAI_API_KEY is missing. "
            "Watermark removal skipped."
        )
        return None

    try:
        source = _prepare_image(image_bytes)

        # ----------------------------------------------------
        # LOCAL DETECTION
        # ----------------------------------------------------

        watermark_mask = _create_watermark_mask(source)

        if not _mask_has_enough_pixels(watermark_mask):
            logger.info(
                "✅ No convincing watermark detected. "
                "Skipping OpenAI request."
            )
            return None

        # ----------------------------------------------------
        # CONVERT IMAGE + MASK TO PNG FILE OBJECTS
        # ----------------------------------------------------

        image_buffer = io.BytesIO()

        source.save(
            image_buffer,
            format="PNG",
        )

        image_buffer.seek(0)
        image_buffer.name = "input.png"

        mask_buffer = io.BytesIO()

        watermark_mask.save(
            mask_buffer,
            format="PNG",
        )

        mask_buffer.seek(0)
        mask_buffer.name = "mask.png"

        # ----------------------------------------------------
        # ONE AI EDIT REQUEST
        # ----------------------------------------------------

        logger.info(
            "🎨 Sending image to OpenAI for watermark removal..."
        )

        logger.info(
            "🤖 Model: %s",
            OPENAI_IMAGE_MODEL,
        )

        logger.info(
            "💰 Quality: %s",
            OPENAI_IMAGE_QUALITY,
        )

        response = await client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=image_buffer,
            mask=mask_buffer,
            prompt=(
                "Remove only the visible @cappersfree watermark "
                "and CF logo/watermark from the masked areas. "
                "Reconstruct the pixels behind the watermark so "
                "they naturally match the surrounding image. "
                "Preserve the original subject, people, faces, "
                "objects, text, colors, lighting, composition, "
                "background, framing, and image quality. "
                "Do not crop or resize the composition. "
                "Do not create new objects. "
                "Do not change anything outside the masked "
                "watermark areas."
            ),
            quality=OPENAI_IMAGE_QUALITY,
        )

        # ----------------------------------------------------
        # GET OUTPUT
        # ----------------------------------------------------

        if not response.data:
            logger.error(
                "❌ OpenAI returned no image data."
            )
            return None

        result = response.data[0]

        b64_json = getattr(
            result,
            "b64_json",
            None,
        )

        if not b64_json:
            logger.error(
                "❌ OpenAI response did not contain b64_json."
            )
            return None

        cleaned_bytes = base64.b64decode(
            b64_json
        )

        if not cleaned_bytes:
            logger.error(
                "❌ Decoded edited image is empty."
            )
            return None

        logger.info(
            "✅ Watermark removal successful: %d bytes",
            len(cleaned_bytes),
        )

        return cleaned_bytes

    except Exception:
        logger.exception(
            "❌ OpenAI watermark removal failed."
        )
        return None


# ============================================================
# COMPATIBILITY FUNCTION
# ============================================================

async def regenerate_image_from_bytes(
    image_bytes: bytes,
    filename: str = "image.jpg",
):
    """
    Compatibility wrapper for existing main.py.
    """

    return await remove_watermarks_from_bytes(
        image_bytes,
        filename,
    )
