import os
import re
import base64
import time
import asyncio
import logging
import subprocess
import tempfile

from pathlib import Path
from io import BytesIO

from PIL import Image
from openai import OpenAI
from aiogram.types import BufferedInputFile


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger("ai_processor")


# ============================================================
# OPENAI
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    logger.warning("⚠️ OPENAI_API_KEY is missing!")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# OPENAI MODELS
# ============================================================

# Compatible with the pinned OpenAI Python SDK.
VISION_MODEL = "gpt-4o-mini"

# Image generation model.
IMAGE_MODEL = "gpt-image-1"


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
# OPENAI COOLDOWNS
# ============================================================

_last_vision_call = 0.0
_vision_cooldown = 3.0

_last_generation_call = 0.0
_generation_cooldown = 3.0


# ============================================================
# STARTUP CONFIG LOG
# ============================================================

logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
logger.info("🤖 AI PROCESSOR STARTED")
logger.info("👤 OLD_MENTION: %r", OLD_MENTION)
logger.info("👤 NEW_MENTION: %r", NEW_MENTION)
logger.info("🧠 Caption AI rewriting: DISABLED")
logger.info("✏️ Caption processing: EXACT REPLACEMENT")
logger.info(
    "🖼️ Image pipeline: VISION → DECONSTRUCTION → GENERATION"
)
logger.info("👁️ Vision model: %s", VISION_MODEL)
logger.info("🎨 Image model: %s", IMAGE_MODEL)
logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ============================================================
# CAPTION PROCESSING
# ============================================================

def replace_username(text: str) -> str:
    """
    Only:

    1. Remove every literal '*'
    2. Replace OLD_MENTION with NEW_MENTION

    No AI rewriting.
    No paraphrasing.
    """

    if not text:
        return text

    result = text.replace("*", "")

    if NEW_MENTION:

        pattern = re.escape(OLD_MENTION)

        result = re.sub(
            pattern,
            NEW_MENTION,
            result,
            flags=re.IGNORECASE,
        )

    else:

        logger.warning(
            "⚠️ NEW_MENTION is empty. "
            "Username was not replaced."
        )

    return result


async def rewrite_text(
    original_text: str,
) -> str:
    """
    Kept with the same function name so main.py
    does not need to change.

    NO OpenAI.
    NO DeepSeek.
    NO paraphrasing.

    Only removes '*' and replaces the configured username.
    """

    result = replace_username(original_text)

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("📝 CAPTION PROCESSING")
    logger.info("📝 ORIGINAL: %r", original_text)
    logger.info("📝 FINAL:    %r", result)
    logger.info("👤 OLD:      %r", OLD_MENTION)
    logger.info("👤 NEW:      %r", NEW_MENTION)
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return result


# ============================================================
# IMAGE PREPARATION
# ============================================================

def resize_for_vision(
    img: Image.Image,
) -> Image.Image:
    """
    Resize while preserving aspect ratio.
    """

    max_dimension = 2048

    if max(img.size) <= max_dimension:
        return img

    scale = max_dimension / max(img.size)

    new_size = (
        max(
            1,
            int(img.width * scale),
        ),
        max(
            1,
            int(img.height * scale),
        ),
    )

    return img.resize(
        new_size,
        Image.Resampling.LANCZOS,
    )


def _is_video_container(data: bytes) -> bool:
    """
    Detect MP4/MOV/WebM containers.
    """

    if len(data) >= 12 and data[4:8] == b"ftyp":
        return True

    return data.startswith(
        b"\x1a\x45\xdf\xa3"
    )


# ============================================================
# CONTACT SHEET
# ============================================================

def _create_contact_sheet(
    frames,
):
    """
    Create a contact sheet from PIL frames.
    """

    if not frames:
        return None

    columns = min(3, len(frames))
    rows = (
        len(frames) + columns - 1
    ) // columns

    cell_width = max(
        frame.width
        for frame in frames
    )

    cell_height = max(
        frame.height
        for frame in frames
    )

    padding = 20

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width
            + (columns + 1) * padding,

            rows * cell_height
            + (rows + 1) * padding,
        ),
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


# ============================================================
# VIDEO / GIF EXTRACTION
# ============================================================

def _extract_video_contact_sheet(
    video_bytes: bytes,
):
    """
    Telegram GIFs are frequently MP4 files.

    Extract representative frames using
    imageio-ffmpeg.
    """

    try:

        from imageio_ffmpeg import (
            get_ffmpeg_exe,
        )

    except ImportError:

        logger.error(
            "❌ imageio-ffmpeg is missing. "
            "Add imageio-ffmpeg to requirements.txt."
        )

        return None

    try:

        ffmpeg = get_ffmpeg_exe()

        logger.info(
            "🎞️ FFmpeg executable: %s",
            ffmpeg,
        )

        with tempfile.TemporaryDirectory() as tmp:

            input_path = (
                Path(tmp)
                / "telegram_media.mp4"
            )

            output_pattern = str(
                Path(tmp)
                / "frame_%02d.jpg"
            )

            input_path.write_bytes(
                video_bytes
            )

            command = [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vf",
                (
                    "fps=1/1,"
                    "scale=1536:-2:"
                    "force_original_aspect_ratio=decrease"
                ),
                "-frames:v",
                "6",
                output_pattern,
            ]

            logger.info(
                "🎞️ Running FFmpeg..."
            )

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=90,
            )

            if result.returncode != 0:

                logger.error(
                    "❌ FFmpeg failed: %s",
                    result.stderr.decode(
                        "utf-8",
                        errors="replace",
                    )[-3000:],
                )

                return None

            frame_paths = sorted(
                Path(tmp).glob(
                    "frame_*.jpg"
                )
            )

            logger.info(
                "🎞️ FFmpeg produced %d frame(s).",
                len(frame_paths),
            )

            if not frame_paths:

                logger.error(
                    "❌ FFmpeg produced no frames."
                )

                return None

            frames = []

            for frame_path in frame_paths:

                try:

                    with Image.open(
                        frame_path
                    ) as frame:

                        prepared = (
                            resize_for_vision(
                                frame.convert(
                                    "RGB"
                                )
                            ).copy()
                        )

                        frames.append(
                            prepared
                        )

                except Exception:

                    logger.exception(
                        "⚠️ Could not read "
                        "extracted frame %s",
                        frame_path,
                    )

            if not frames:

                logger.error(
                    "❌ No usable video frames."
                )

                return None

            logger.info(
                "🎞️ Successfully decoded %d frame(s).",
                len(frames),
            )

            sheet = _create_contact_sheet(
                frames
            )

            if sheet is not None:

                logger.info(
                    "✅ Video/GIF contact sheet created: %sx%s",
                    sheet.width,
                    sheet.height,
                )

            return sheet

    except Exception:

        logger.exception(
            "❌ VIDEO/GIF FRAME EXTRACTION FAILED"
        )

        return None


# ============================================================
# IMAGE / GIF PREPARATION
# ============================================================

def prepare_image_for_vision(
    image_bytes: bytes,
):
    """
    Prepare:

    - JPG
    - JPEG
    - PNG
    - WEBP
    - GIF
    - animated GIF
    - Telegram GIF-as-MP4
    - MP4
    - MOV
    - WebM

    for Vision.
    """

    if not image_bytes:

        logger.error(
            "❌ prepare_image_for_vision "
            "received empty bytes."
        )

        return None

    logger.info(
        "🔬 Media signature: %s",
        bytes(
            image_bytes[:32]
        ).hex(" "),
    )

    logger.info(
        "📦 Media size: %d bytes",
        len(image_bytes),
    )

    try:

        # ----------------------------------------------------
        # VIDEO / MP4 / MOV / WEBM
        # ----------------------------------------------------

        if _is_video_container(
            image_bytes
        ):

            logger.info(
                "🎞️ MP4/MOV/WebM container detected."
            )

            logger.info(
                "🎞️ Preparing video/GIF media with FFmpeg..."
            )

            return _extract_video_contact_sheet(
                image_bytes
            )

        # ----------------------------------------------------
        # NORMAL IMAGE / GIF
        # ----------------------------------------------------

        source = Image.open(
            BytesIO(image_bytes)
        )

        source.load()

        # ----------------------------------------------------
        # STATIC IMAGE
        # ----------------------------------------------------

        if not bool(
            getattr(
                source,
                "is_animated",
                False,
            )
        ):

            rgba = source.convert(
                "RGBA"
            )

            background = Image.new(
                "RGB",
                rgba.size,
                "white",
            )

            background.paste(
                rgba,
                (0, 0),
                rgba.getchannel("A"),
            )

            return resize_for_vision(
                background
            )

        # ----------------------------------------------------
        # ANIMATED GIF
        # ----------------------------------------------------

        total_frames = int(
            getattr(
                source,
                "n_frames",
                1,
            )
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
                    i
                    * (total_frames - 1)
                    / (sample_count - 1)
                )
                for i in range(
                    sample_count
                )
            ]

        frames = []

        for frame_number in frame_indexes:

            try:

                source.seek(
                    frame_number
                )

                rgba = source.convert(
                    "RGBA"
                )

                background = Image.new(
                    "RGB",
                    rgba.size,
                    "white",
                )

                background.paste(
                    rgba,
                    (0, 0),
                    rgba.getchannel("A"),
                )

                frames.append(
                    resize_for_vision(
                        background
                    ).copy()
                )

            except Exception:

                logger.exception(
                    "⚠️ Could not read GIF frame %s",
                    frame_number,
                )

        if not frames:

            logger.error(
                "❌ No GIF frames could be extracted."
            )

            return None

        logger.info(
            "🎞️ Animated GIF detected: "
            "%d total frames; using %d representative frames.",
            total_frames,
            len(frames),
        )

        return _create_contact_sheet(
            frames
        )

    except Exception:

        logger.exception(
            "❌ IMAGE PREPARATION FAILED"
        )

        return None


# ============================================================
# OPENAI VISION
# ============================================================

async def describe_image_bytes(
    image_bytes: bytes,
):
    """
    Send the prepared image to OpenAI Vision.

    IMPORTANT:
    This intentionally uses:

        client.chat.completions.create()

    and NOT:

        client.responses.create()

    because the application is pinned to:
        openai==1.58.0
    """

    global _last_vision_call

    if not image_bytes:

        logger.error(
            "❌ Vision received empty image bytes."
        )

        return None

    # --------------------------------------------------------
    # COOLDOWN
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - _last_vision_call
    )

    if elapsed < _vision_cooldown:

        wait = (
            _vision_cooldown
            - elapsed
        )

        await asyncio.sleep(
            wait
        )

    _last_vision_call = time.time()

    # --------------------------------------------------------
    # PREPARE IMAGE
    # --------------------------------------------------------

    try:

        image = prepare_image_for_vision(
            image_bytes
        )

        if image is None:

            return None

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=92,
            optimize=True,
        )

        prepared_bytes = (
            buffer.getvalue()
        )

        encoded = base64.b64encode(
            prepared_bytes
        ).decode("utf-8")

        image_url = (
            "data:image/jpeg;base64,"
            + encoded
        )

        logger.info(
            "✅ Vision JPEG prepared: %d bytes",
            len(prepared_bytes),
        )

    except Exception as e:

        logger.exception(
            "❌ Failed to prepare Vision image: %s",
            e,
        )

        return None

    # --------------------------------------------------------
    # VISION PROMPT
    # --------------------------------------------------------

    prompt = """
Analyze this image carefully and create a detailed
reconstruction specification for a NEW image.

The purpose is to understand the visual design and legitimate
content so another image-generation model can create a new
similar graphic.

IMPORTANT:

Do NOT reproduce:
- watermarks
- usernames
- @handles
- social-media handles
- channel branding
- third-party logos
- third-party brand marks

Describe those areas generically instead.

Do NOT include the actual watermark or username in your
reconstruction specification.

============================================================
1. CANVAS
============================================================

Describe:
- orientation
- approximate aspect ratio
- layout dimensions
- overall proportions

============================================================
2. MAIN CONTENT
============================================================

Describe all legitimate visible content:

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

============================================================
3. LEGITIMATE TEXT
============================================================

Transcribe visible legitimate text accurately.

Include:
- headings
- matchups
- scores
- odds
- picks
- dates
- times
- labels
- numbers

Do NOT include:
- usernames
- @handles
- watermarks
- channel names
- branding

Never invent text that is not visible.

============================================================
4. COMPOSITION
============================================================

Describe:
- top section
- center section
- bottom section
- left/right placement
- alignment
- spacing
- margins
- cards
- panels
- borders
- dividers

============================================================
5. TYPOGRAPHY
============================================================

Describe:
- font style
- font weight
- approximate size hierarchy
- uppercase/lowercase
- alignment
- text color
- outlines
- shadows
- glow
- spacing

============================================================
6. COLORS
============================================================

Describe:
- background
- primary colors
- secondary colors
- accent colors
- gradients
- highlights
- shadows

============================================================
7. BACKGROUND
============================================================

Describe:
- texture
- lighting
- atmosphere
- depth
- blur
- patterns
- gradients
- shadows
- environmental details

============================================================
8. GRAPHIC ELEMENTS
============================================================

Describe legitimate:
- icons
- arrows
- shapes
- lines
- badges
- panels
- borders
- decorative elements

============================================================
9. ANIMATION
============================================================

If the image is a contact sheet from a GIF:

Describe:
- what changes between frames
- movement
- transitions
- which elements remain fixed
- which elements change
- overall animation concept

============================================================
10. FINAL GENERATION SPECIFICATION
============================================================

Finish with a detailed specification that another image
generator can use to create a NEW image with:

- similar composition
- similar hierarchy
- similar visual style
- similar color relationships
- similar legitimate information
- similar overall design language

while excluding watermarks, usernames, @handles,
channel branding and third-party logos.

Do not invent information.
"""

    # --------------------------------------------------------
    # VISION REQUEST
    # --------------------------------------------------------

    max_retries = 3

    for attempt in range(
        max_retries
    ):

        try:

            logger.info(
                "🔍 OpenAI Vision request %d/%d",
                attempt + 1,
                max_retries,
            )

            # =================================================
            # IMPORTANT:
            # openai==1.58.0 compatible API
            #
            # NO client.responses
            # =================================================

            response = (
                openai_client
                .chat
                .completions
                .create(
                    model=VISION_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
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
            )

            if not response:

                logger.error(
                    "❌ Vision returned no response."
                )

                return None

            if not response.choices:

                logger.error(
                    "❌ Vision returned no choices."
                )

                return None

            description = (
                response
                .choices[0]
                .message
                .content
            )

            if not description:

                logger.error(
                    "❌ Vision returned empty text."
                )

                return None

            description = description.strip()

            logger.info(
                "✅ Vision deconstruction complete (%d chars)",
                len(description),
            )

            logger.info(
                "🧩 Vision preview:\n%s",
                description[:1200],
            )

            return description

        except Exception as e:

            status_code = getattr(
                e,
                "status_code",
                None,
            )

            logger.error(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            logger.error(
                "❌ OPENAI VISION ERROR"
            )

            logger.error(
                "❌ Type: %s",
                type(e).__name__,
            )

            logger.error(
                "❌ Error: %s",
                str(e),
            )

            logger.error(
                "❌ Status: %s",
                status_code,
            )

            body = getattr(
                e,
                "body",
                None,
            )

            if body:

                logger.error(
                    "❌ API ERROR BODY: %s",
                    body,
                )

            logger.error(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            if (
                status_code == 429
                and attempt < max_retries - 1
            ):

                wait = 2 ** attempt

                logger.warning(
                    "⏳ Retrying Vision in %ss...",
                    wait,
                )

                await asyncio.sleep(
                    wait
                )

                continue

            return None

    return None


# ============================================================
# IMAGE GENERATION
# ============================================================

async def generate_image_from_description(
    description: str,
):
    """
    Generate a NEW image from the Vision
    reconstruction specification.
    """

    global _last_generation_call

    if not description:

        logger.error(
            "❌ Empty reconstruction description."
        )

        return None

    # --------------------------------------------------------
    # COOLDOWN
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - _last_generation_call
    )

    if elapsed < _generation_cooldown:

        wait = (
            _generation_cooldown
            - elapsed
        )

        await asyncio.sleep(
            wait
        )

    _last_generation_call = time.time()

    # --------------------------------------------------------
    # GENERATION PROMPT
    # --------------------------------------------------------

    generation_prompt = f"""
Create a NEW professional sports graphic using the
reconstruction specification below.

This is a NEW generated image, not an edited copy.

============================================================
DESIGN
============================================================

Preserve the described:
- composition
- hierarchy
- spacing
- visual structure
- colors
- typography style
- background treatment
- legitimate information

============================================================
TEXT
============================================================

Make legitimate text as accurate and readable as possible.

Do not invent:
- scores
- odds
- dates
- teams
- players
- picks
- numbers

============================================================
EXCLUDE
============================================================

Do NOT include:
- watermarks
- usernames
- @handles
- social-media handles
- channel branding
- third-party logos
- third-party brand marks

Any area that previously contained such material should
become a clean part of the new design.

============================================================
QUALITY
============================================================

Create:
- sharp text
- professional typography
- clean spacing
- strong hierarchy
- polished sports-graphic appearance
- coherent colors
- clean background
- high visual quality

============================================================
RECONSTRUCTION SPECIFICATION
============================================================

{description}
"""

    # --------------------------------------------------------
    # REQUEST
    # --------------------------------------------------------

    try:

        logger.info(
            "🎨 Sending reconstruction specification "
            "to image generation..."
        )

        logger.info(
            "📝 Prompt length: %d characters",
            len(generation_prompt),
        )

        response = (
            openai_client
            .images
            .generate(
                model=IMAGE_MODEL,
                prompt=generation_prompt,
                n=1,
            )
        )

        if not response:

            logger.error(
                "❌ OpenAI returned no response."
            )

            return None

        if not response.data:

            logger.error(
                "❌ OpenAI returned no image data."
            )

            return None

        image_data = response.data[0]

        # ----------------------------------------------------
        # BASE64
        # ----------------------------------------------------

        b64_json = getattr(
            image_data,
            "b64_json",
            None,
        )

        if b64_json:

            image_bytes = (
                base64.b64decode(
                    b64_json
                )
            )

            logger.info(
                "✅ NEW IMAGE GENERATED (%d bytes)",
                len(image_bytes),
            )

            return BufferedInputFile(
                file=image_bytes,
                filename="regenerated.png",
            )

        # ----------------------------------------------------
        # URL
        # ----------------------------------------------------

        image_url = getattr(
            image_data,
            "url",
            None,
        )

        if image_url:

            logger.info(
                "✅ NEW IMAGE GENERATED as URL."
            )

            return image_url

        logger.error(
            "❌ OpenAI response contained "
            "neither b64_json nor URL."
        )

        return None

    except Exception as e:

        logger.error(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        logger.error(
            "❌ OPENAI IMAGE GENERATION ERROR"
        )

        logger.error(
            "❌ Exception type: %s",
            type(e).__name__,
        )

        logger.error(
            "❌ Exception: %s",
            str(e),
        )

        logger.error(
            "❌ Status code: %s",
            getattr(
                e,
                "status_code",
                None,
            ),
        )

        logger.error(
            "❌ Error code: %s",
            getattr(
                e,
                "code",
                None,
            ),
        )

        logger.error(
            "❌ Parameter: %s",
            getattr(
                e,
                "param",
                None,
            ),
        )

        body = getattr(
            e,
            "body",
            None,
        )

        if body:

            logger.error(
                "❌ API ERROR BODY: %s",
                body,
            )

        logger.error(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        return None


# ============================================================
# COMPLETE REGENERATION PIPELINE
# ============================================================

async def regenerate_image_from_bytes(
    image_bytes: bytes,
):
    """
    Complete pipeline:

        Telegram image/GIF
                ↓
        Prepare image
                ↓
        OpenAI Vision
                ↓
        Deconstruction
                ↓
        OpenAI image generation
                ↓
        NEW IMAGE
    """

    if not image_bytes:

        logger.error(
            "❌ regenerate_image_from_bytes "
            "received no bytes."
        )

        return None

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    logger.info(
        "🖼️ IMAGE RECREATION PIPELINE START"
    )

    logger.info(
        "📦 Original bytes: %d",
        len(image_bytes),
    )

    logger.info(
        "🔬 Original signature: %s",
        bytes(
            image_bytes[:32]
        ).hex(" "),
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    # ========================================================
    # STEP 1 — VISION
    # ========================================================

    logger.info(
        "1️⃣ STEP 1/2 — OpenAI Vision deconstruction..."
    )

    try:

        description = (
            await describe_image_bytes(
                image_bytes
            )
        )

    except Exception as e:

        logger.exception(
            "❌ VISION PIPELINE CRASHED: %s",
            e,
        )

        return None

    if not description:

        logger.error(
            "❌ STEP 1 FAILED."
        )

        logger.error(
            "❌ No reconstruction specification "
            "was returned."
        )

        return None

    # ========================================================
    # STEP 2 — GENERATION
    # ========================================================

    logger.info(
        "2️⃣ STEP 2/2 — OpenAI image generation..."
    )

    try:

        generated = (
            await generate_image_from_description(
                description
            )
        )

    except Exception as e:

        logger.exception(
            "❌ GENERATION PIPELINE CRASHED: %s",
            e,
        )

        return None

    if not generated:

        logger.error(
            "❌ STEP 2 FAILED."
        )

        logger.error(
            "❌ No regenerated image was produced."
        )

        return None

    logger.info(
        "✅ NEW IMAGE CREATED"
    )

    logger.info(
        "✅ IMAGE RECREATION PIPELINE COMPLETE"
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return generated
