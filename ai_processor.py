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
from typing import Optional, Union

from PIL import Image
from openai import OpenAI
from aiogram.types import BufferedInputFile


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("ai_processor")


# ============================================================
# OPENAI CONFIGURATION
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY is missing!")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
)


# Vision model used to analyze the original/contact sheet.
#
# You can override this in Render:
#
# OPENAI_VISION_MODEL=gpt-4o-mini
#
OPENAI_VISION_MODEL = os.getenv(
    "OPENAI_VISION_MODEL",
    "gpt-4o-mini",
).strip()


# Image generation model.
#
# Current configuration:
#
# OPENAI_IMAGE_MODEL=gpt-image-2
#
OPENAI_IMAGE_MODEL = os.getenv(
    "OPENAI_IMAGE_MODEL",
    "gpt-image-2",
).strip()


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
_vision_cooldown = float(
    os.getenv(
        "VISION_COOLDOWN",
        "3",
    )
)

_last_generation_call = 0.0
_generation_cooldown = float(
    os.getenv(
        "GENERATION_COOLDOWN",
        "3",
    )
)


# ============================================================
# STARTUP LOG
# ============================================================

logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
logger.info("🤖 AI PROCESSOR STARTED")
logger.info(
    "👤 OLD_MENTION: %r",
    OLD_MENTION,
)
logger.info(
    "👤 NEW_MENTION: %r",
    NEW_MENTION,
)
logger.info(
    "🧠 Vision model: %s",
    OPENAI_VISION_MODEL,
)
logger.info(
    "🎨 Image model: %s",
    OPENAI_IMAGE_MODEL,
)
logger.info(
    "📝 Caption AI rewriting: DISABLED",
)
logger.info(
    "✏️ Caption processing: EXACT REPLACEMENT",
)
logger.info(
    "🖼️ Image pipeline: VISION → DECONSTRUCTION → GENERATION",
)
logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ============================================================
# CAPTION PROCESSING
# ============================================================

def replace_username(text: str) -> str:
    """
    Process Telegram caption.

    ONLY:
        1. Remove literal '*'
        2. Replace OLD_MENTION with NEW_MENTION

    No AI rewriting.
    No paraphrasing.
    No other text modification.
    """

    if not text:
        return text

    result = text.replace("*", "")

    if NEW_MENTION:

        pattern = re.escape(
            OLD_MENTION
        )

        result = re.sub(
            pattern,
            NEW_MENTION,
            result,
            flags=re.IGNORECASE,
        )

    else:

        logger.warning(
            "⚠️ NEW_MENTION is empty. "
            "Username replacement skipped."
        )

    return result


async def rewrite_text(
    original_text: str,
) -> str:
    """
    Kept with the original function name so main.py
    does not need to change.

    This is NOT AI rewriting.

    Only:
        remove '*'
        replace OLD_MENTION
    """

    result = replace_username(
        original_text
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    logger.info(
        "📝 CAPTION PROCESSING"
    )

    logger.info(
        "📝 ORIGINAL: %r",
        original_text,
    )

    logger.info(
        "📝 FINAL:    %r",
        result,
    )

    logger.info(
        "👤 OLD:      %r",
        OLD_MENTION,
    )

    logger.info(
        "👤 NEW:      %r",
        NEW_MENTION,
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return result


# ============================================================
# IMAGE PREPARATION
# ============================================================

def resize_for_vision(
    img: Image.Image,
) -> Image.Image:
    """
    Resize while preserving aspect ratio.

    Maximum dimension = 2048 px.
    """

    max_dimension = 2048

    if max(img.size) <= max_dimension:
        return img

    scale = (
        max_dimension
        / max(img.size)
    )

    new_size = (
        max(
            1,
            int(
                img.width * scale
            ),
        ),
        max(
            1,
            int(
                img.height * scale
            ),
        ),
    )

    return img.resize(
        new_size,
        Image.Resampling.LANCZOS,
    )


def _is_video_container(
    data: bytes,
) -> bool:
    """
    Detect MP4/MOV/WebM containers.
    """

    if (
        len(data) >= 12
        and data[4:8] == b"ftyp"
    ):
        return True

    return data.startswith(
        b"\x1a\x45\xdf\xa3"
    )


# ============================================================
# VIDEO / GIF CONTACT SHEET
# ============================================================

def _extract_video_contact_sheet(
    video_bytes: bytes,
):
    """
    Telegram GIFs are frequently delivered as MP4.

    Extract representative frames using FFmpeg.
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
                / "telegram_media"
            )

            output_pattern = (
                str(
                    Path(tmp)
                    / "frame_%02d.jpg"
                )
            )

            input_path.write_bytes(
                video_bytes
            )

            command = [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),

                # One frame every second.
                "-vf",
                (
                    "fps=1/1,"
                    "scale=1536:-2:"
                    "force_original_aspect_ratio=decrease"
                ),

                "-frames:v",
                "6",

                "-q:v",
                "2",

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

                stderr = (
                    result.stderr
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )

                logger.error(
                    "❌ FFmpeg failed: %s",
                    stderr[-3000:],
                )

                return None

            frame_paths = sorted(
                Path(tmp).glob(
                    "frame_*.jpg"
                )
            )

            if not frame_paths:

                logger.error(
                    "❌ FFmpeg produced no frames."
                )

                return None

            logger.info(
                "🎞️ FFmpeg produced %d frame(s).",
                len(frame_paths),
            )

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
                            )
                        )

                        frames.append(
                            prepared.copy()
                        )

                except Exception:

                    logger.exception(
                        "⚠️ Could not read extracted frame %s",
                        frame_path,
                    )

            if not frames:

                logger.error(
                    "❌ No decoded video frames."
                )

                return None

            logger.info(
                "🎞️ Successfully decoded %d frame(s).",
                len(frames),
            )

            # ------------------------------------------------
            # Build contact sheet.
            # ------------------------------------------------

            columns = min(
                3,
                len(frames),
            )

            rows = (
                len(frames)
                + columns
                - 1
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

            for index, frame in enumerate(
                frames
            ):

                x = (
                    padding
                    + (
                        index % columns
                    )
                    * (
                        cell_width
                        + padding
                    )
                )

                y = (
                    padding
                    + (
                        index // columns
                    )
                    * (
                        cell_height
                        + padding
                    )
                )

                sheet.paste(
                    frame,
                    (
                        x,
                        y,
                    ),
                )

            sheet = resize_for_vision(
                sheet
            )

            logger.info(
                "✅ Video/GIF contact sheet created: %sx%s",
                sheet.width,
                sheet.height,
            )

            return sheet

    except subprocess.TimeoutExpired:

        logger.error(
            "❌ FFmpeg timed out."
        )

        return None

    except Exception:

        logger.exception(
            "❌ VIDEO/GIF FRAME EXTRACTION FAILED"
        )

        return None


# ============================================================
# PIL IMAGE / GIF CONTACT SHEET
# ============================================================

def _prepare_pillow_media(
    image_bytes: bytes,
):
    """
    Prepare normal image or animated GIF.
    """

    source = Image.open(
        BytesIO(image_bytes)
    )

    source.load()

    is_animated = bool(
        getattr(
            source,
            "is_animated",
            False,
        )
    )

    # --------------------------------------------------------
    # Normal image.
    # --------------------------------------------------------

    if not is_animated:

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
            rgba.getchannel(
                "A"
            ),
        )

        return resize_for_vision(
            background
        )

    # --------------------------------------------------------
    # Animated GIF.
    # --------------------------------------------------------

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
                * (
                    total_frames
                    - 1
                )
                / (
                    sample_count
                    - 1
                )
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
                rgba.getchannel(
                    "A"
                ),
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

    columns = min(
        3,
        len(frames),
    )

    rows = (
        len(frames)
        + columns
        - 1
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

    for index, frame in enumerate(
        frames
    ):

        x = (
            padding
            + (
                index % columns
            )
            * (
                cell_width
                + padding
            )
        )

        y = (
            padding
            + (
                index // columns
            )
            * (
                cell_height
                + padding
            )
        )

        sheet.paste(
            frame,
            (
                x,
                y,
            ),
        )

    return resize_for_vision(
        sheet
    )


# ============================================================
# PREPARE MEDIA FOR VISION
# ============================================================

def prepare_image_for_vision(
    image_bytes: bytes,
):
    """
    Prepare:
        PNG
        JPG
        JPEG
        WEBP
        GIF
        MP4
        MOV
        WEBM

    into a JPEG image suitable for Vision.
    """

    if not image_bytes:

        logger.error(
            "❌ prepare_image_for_vision received empty bytes."
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
        # Video / MP4 GIF.
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
        # Pillow image/GIF.
        # ----------------------------------------------------

        return _prepare_pillow_media(
            image_bytes
        )

    except Exception:

        logger.exception(
            "❌ IMAGE PREPARATION FAILED"
        )

        return None


# ============================================================
# VISION PROMPT
# ============================================================

VISION_PROMPT = """
Analyze this image carefully and create a detailed
reconstruction specification for a NEW image.

The purpose is to understand the visual design and legitimate
visible content so another image-generation model can create a
new graphic.

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

Do NOT include the actual watermark or username in the
reconstruction specification.

============================================================
1. CANVAS
============================================================

Describe:
- orientation
- approximate aspect ratio
- overall proportions
- major dimensions

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
- letter spacing

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

If this is a contact sheet from an animated GIF/video:

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

Finish with a detailed specification another image generator
can use to create a NEW image with:

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


# ============================================================
# VISION REQUEST
# ============================================================

async def describe_image_bytes(
    image_bytes: bytes,
):
    """
    Send prepared image to OpenAI Vision.

    Uses the current Responses API.
    """

    global _last_vision_call

    if not image_bytes:

        logger.error(
            "❌ Vision received empty image bytes."
        )

        return None

    # --------------------------------------------------------
    # Cooldown.
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

        logger.info(
            "⏳ Vision cooldown: %.2fs",
            wait,
        )

        await asyncio.sleep(
            wait
        )

    # --------------------------------------------------------
    # Prepare media.
    # --------------------------------------------------------

    try:

        image = prepare_image_for_vision(
            image_bytes
        )

        if image is None:

            logger.error(
                "❌ Could not prepare media for Vision."
            )

            return None

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=92,
            optimize=True,
        )

        jpeg_bytes = (
            buffer.getvalue()
        )

        encoded = base64.b64encode(
            jpeg_bytes
        ).decode(
            "utf-8"
        )

        image_url = (
            "data:image/jpeg;base64,"
            + encoded
        )

        logger.info(
            "✅ Vision JPEG prepared: %d bytes",
            len(jpeg_bytes),
        )

    except Exception:

        logger.exception(
            "❌ Failed to prepare Vision image."
        )

        return None

    # --------------------------------------------------------
    # Ensure SDK supports Responses API.
    # --------------------------------------------------------

    responses_api = getattr(
        openai_client,
        "responses",
        None,
    )

    if responses_api is None:

        logger.error(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        logger.error(
            "❌ OPENAI SDK DOES NOT SUPPORT RESPONSES API"
        )

        logger.error(
            "❌ Installed OpenAI SDK is too old or incompatible."
        )

        logger.error(
            "❌ Upgrade with: pip install -U openai"
        )

        logger.error(
            "❌ Add/update requirements.txt with:"
        )

        logger.error(
            "❌ openai>=1.100.0"
        )

        logger.error(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        return None

    # --------------------------------------------------------
    # Retry.
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

            _last_vision_call = time.time()

            def make_request():

                return openai_client.responses.create(
                    model=OPENAI_VISION_MODEL,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": VISION_PROMPT,
                                },
                                {
                                    "type": "input_image",
                                    "image_url": image_url,
                                    "detail": "high",
                                },
                            ],
                        }
                    ],
                    max_output_tokens=4000,
                )

            response = await asyncio.to_thread(
                make_request
            )

            description = getattr(
                response,
                "output_text",
                None,
            )

            if not description:

                logger.error(
                    "❌ Vision returned empty output_text."
                )

                logger.error(
                    "❌ Response type: %s",
                    type(response).__name__,
                )

                return None

            description = description.strip()

            logger.info(
                "✅ Vision deconstruction complete (%d chars)",
                len(description),
            )

            logger.info(
                "🧩 Vision preview:"
            )

            logger.info(
                "%s",
                description[:1200],
            )

            return description

        except Exception as e:

            status_code = getattr(
                e,
                "status_code",
                None,
            )

            error_code = getattr(
                e,
                "code",
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

            logger.error(
                "❌ Code: %s",
                error_code,
            )

            body = getattr(
                e,
                "body",
                None,
            )

            if body:

                logger.error(
                    "❌ API BODY: %s",
                    body,
                )

            logger.error(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            # Retry rate limits / temporary server errors.
            retryable = (
                status_code == 429
                or (
                    status_code is not None
                    and status_code >= 500
                )
            )

            if (
                retryable
                and attempt < max_retries - 1
            ):

                wait = 2 ** attempt

                logger.warning(
                    "⏳ Retrying Vision in %ds...",
                    wait,
                )

                await asyncio.sleep(
                    wait
                )

                continue

            return None

    return None


# ============================================================
# IMAGE GENERATION PROMPT
# ============================================================

def build_generation_prompt(
    description: str,
) -> str:

    return f"""
Create a NEW professional sports graphic using the
reconstruction specification below.

This is a NEW generated image.

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

Make legitimate visible text as accurate and readable as
possible.

Do not invent:
- scores
- odds
- dates
- teams
- players
- picks
- numbers

Only use information supported by the reconstruction
specification.

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

Any area that previously contained such material should become
a clean, neutral part of the new design.

============================================================
QUALITY
============================================================

Create:
- sharp readable text
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


# ============================================================
# IMAGE GENERATION
# ============================================================

async def generate_image_from_description(
    description: str,
):
    """
    Generate a NEW image from the Vision reconstruction
    specification.
    """

    global _last_generation_call

    if not description:

        logger.error(
            "❌ Empty reconstruction description."
        )

        return None

    # --------------------------------------------------------
    # Cooldown.
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

        logger.info(
            "⏳ Generation cooldown: %.2fs",
            wait,
        )

        await asyncio.sleep(
            wait
        )

    _last_generation_call = time.time()

    generation_prompt = (
        build_generation_prompt(
            description
        )
    )

    logger.info(
        "🎨 Sending reconstruction "
        "specification to image generation..."
    )

    logger.info(
        "📝 Generation prompt length: %d characters",
        len(generation_prompt),
    )

    # --------------------------------------------------------
    # Request.
    # --------------------------------------------------------

    try:

        def make_request():

            return openai_client.images.generate(
                model=OPENAI_IMAGE_MODEL,
                prompt=generation_prompt,
                n=1,
            )

        response = await asyncio.to_thread(
            make_request
        )

        if not response:

            logger.error(
                "❌ OpenAI returned no image response."
            )

            return None

        data = getattr(
            response,
            "data",
            None,
        )

        if not data:

            logger.error(
                "❌ OpenAI returned no image data."
            )

            return None

        image_data = data[0]

        # ----------------------------------------------------
        # Base64 response.
        # ----------------------------------------------------

        b64_json = getattr(
            image_data,
            "b64_json",
            None,
        )

        if b64_json:

            image_bytes = base64.b64decode(
                b64_json
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
        # URL response.
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
            "❌ Image response contained neither "
            "b64_json nor URL."
        )

        logger.error(
            "❌ Image response object: %r",
            image_data,
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
# COMPLETE IMAGE REGENERATION PIPELINE
# ============================================================

async def regenerate_image_from_bytes(
    image_bytes: bytes,
):
    """
    Complete pipeline:

        Telegram image/GIF
                ↓
        Prepare media
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

    except Exception:

        logger.exception(
            "❌ VISION PIPELINE CRASHED"
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

    logger.info(
        "✅ STEP 1 COMPLETE"
    )

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

    except Exception:

        logger.exception(
            "❌ GENERATION PIPELINE CRASHED"
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
