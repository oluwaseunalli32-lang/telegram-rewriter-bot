import os
import re
import base64
import time
import asyncio
import subprocess
import tempfile
import logging
from pathlib import Path
from io import BytesIO

from PIL import Image, ImageFile
from openai import OpenAI
from aiogram.types import BufferedInputFile


# ============================================================
# PIL CONFIGURATION
# ============================================================

# Allow Pillow to handle slightly damaged/truncated Telegram
# media instead of immediately rejecting it.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("ai_processor")

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    )
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


# ============================================================
# OPENAI
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY is missing!")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Vision / analysis model.
#
# Keep this configurable from Render so you can change it
# without editing the source code.
VISION_MODEL = os.getenv(
    "OPENAI_VISION_MODEL",
    "gpt-4o-mini",
).strip()


# Current OpenAI image-generation model.
GENERATION_MODEL = os.getenv(
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
_vision_cooldown = 3

_last_generation_call = 0.0
_generation_cooldown = 3


# ============================================================
# STARTUP
# ============================================================

logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
logger.info("🤖 AI PROCESSOR STARTED")
logger.info("👤 OLD_MENTION: %r", OLD_MENTION)
logger.info("👤 NEW_MENTION: %r", NEW_MENTION)
logger.info("🧠 Vision model: %s", VISION_MODEL)
logger.info("🎨 Image model: %s", GENERATION_MODEL)
logger.info("📝 Caption AI rewriting: DISABLED")
logger.info("✏️ Caption processing: EXACT REPLACEMENT")
logger.info("🖼️ Image pipeline: VISION → DECONSTRUCTION → GENERATION")
logger.info("🎞️ MP4/GIF support: ENABLED")
logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ============================================================
# CAPTION PROCESSING
# ============================================================

def replace_username(text: str) -> str:
    """
    ONLY:

    1. Remove literal '*'
    2. Replace OLD_MENTION with NEW_MENTION

    No AI rewriting.
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
            "⚠️ NEW_MENTION is empty. Username was not replaced."
        )

    return result


async def rewrite_text(
    original_text: str,
) -> str:

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
# MEDIA SIGNATURE DETECTION
# ============================================================

def get_media_signature(data: bytes) -> str:
    if not data:
        return "EMPTY"

    return bytes(data[:32]).hex(" ")


def _is_jpeg(data: bytes) -> bool:
    return data.startswith(b"\xff\xd8\xff")


def _is_png(data: bytes) -> bool:
    return data.startswith(b"\x89PNG\r\n\x1a\n")


def _is_gif(data: bytes) -> bool:
    return data.startswith(b"GIF87a") or data.startswith(b"GIF89a")


def _is_webp(data: bytes) -> bool:
    return (
        len(data) >= 12
        and data[:4] == b"RIFF"
        and data[8:12] == b"WEBP"
    )


def _is_video_container(data: bytes) -> bool:
    """
    Detect MP4/MOV and WebM/Matroska.

    MP4 commonly starts with:

        00 00 00 xx 66 74 79 70

    Telegram GIFs frequently arrive this way.
    """

    if len(data) >= 12 and data[4:8] == b"ftyp":
        return True

    # WebM / Matroska EBML signature
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return True

    return False


# ============================================================
# IMAGE RESIZE
# ============================================================

def resize_for_vision(
    img: Image.Image,
) -> Image.Image:

    max_dimension = 2048

    if max(img.size) <= max_dimension:
        return img

    scale = max_dimension / max(img.size)

    new_size = (
        max(1, int(img.width * scale)),
        max(1, int(img.height * scale)),
    )

    return img.resize(
        new_size,
        Image.Resampling.LANCZOS,
    )


# ============================================================
# SAFE RGB CONVERSION
# ============================================================

def image_to_rgb(
    source: Image.Image,
) -> Image.Image:

    """
    Convert any Pillow image to RGB.

    Handles:
    - RGB
    - RGBA
    - palette transparency
    - grayscale
    - CMYK
    """

    try:

        if source.mode == "RGB":
            return source.copy()

        if source.mode in ("RGBA", "LA"):

            rgba = source.convert("RGBA")

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

            return background

        if source.mode == "P":

            rgba = source.convert("RGBA")

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

            return background

        return source.convert("RGB")

    except Exception:

        logger.exception(
            "❌ RGB conversion failed."
        )

        return source.convert("RGB")


# ============================================================
# CONTACT SHEET
# ============================================================

def create_contact_sheet(
    frames,
) -> Image.Image:

    if not frames:
        raise ValueError(
            "No frames supplied to contact sheet."
        )

    columns = min(3, len(frames))
    rows = (len(frames) + columns - 1) // columns

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
# MP4 / VIDEO FRAME EXTRACTION
# ============================================================

def _extract_video_contact_sheet(
    video_bytes: bytes,
):

    logger.info(
        "🎞️ Preparing video/GIF media with FFmpeg..."
    )

    try:

        from imageio_ffmpeg import get_ffmpeg_exe

    except ImportError:

        logger.exception(
            "❌ imageio-ffmpeg is not installed."
        )

        return None

    try:

        ffmpeg = get_ffmpeg_exe()

        logger.info(
            "🎞️ FFmpeg executable: %s",
            ffmpeg,
        )

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = Path(tmp)

            input_path = (
                tmp_path
                / "telegram_media.mp4"
            )

            output_pattern = str(
                tmp_path
                / "frame_%02d.jpg"
            )

            input_path.write_bytes(
                video_bytes
            )

            logger.info(
                "🎞️ Temporary video size: %d bytes",
                len(video_bytes),
            )

            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),

                # Sample approximately 6 frames.
                "-vf",
                (
                    "fps=1/1,"
                    "scale=1536:1536:"
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

            stderr_text = result.stderr.decode(
                "utf-8",
                errors="replace",
            )

            if result.returncode != 0:

                logger.error(
                    "❌ FFmpeg failed. Exit code=%s",
                    result.returncode,
                )

                logger.error(
                    "❌ FFmpeg stderr:\n%s",
                    stderr_text[-5000:],
                )

                return None

            frame_paths = sorted(
                tmp_path.glob("frame_*.jpg")
            )

            if not frame_paths:

                logger.error(
                    "❌ FFmpeg completed but produced no frames."
                )

                if stderr_text:
                    logger.error(
                        "FFmpeg output:\n%s",
                        stderr_text[-3000:],
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

                        frame.load()

                        rgb = image_to_rgb(
                            frame
                        )

                        rgb = resize_for_vision(
                            rgb
                        )

                        frames.append(
                            rgb.copy()
                        )

                except Exception:

                    logger.exception(
                        "⚠️ Failed reading extracted frame: %s",
                        frame_path,
                    )

            if not frames:

                logger.error(
                    "❌ No extracted frames could be opened."
                )

                return None

            logger.info(
                "🎞️ Successfully decoded %d frame(s).",
                len(frames),
            )

            sheet = create_contact_sheet(
                frames
            )

            logger.info(
                "✅ Video/GIF contact sheet created: %sx%s",
                sheet.width,
                sheet.height,
            )

            return sheet

    except subprocess.TimeoutExpired:

        logger.exception(
            "❌ FFmpeg timed out after 90 seconds."
        )

        return None

    except Exception:

        logger.exception(
            "❌ VIDEO/GIF FRAME EXTRACTION FAILED"
        )

        return None


# ============================================================
# GIF FRAME EXTRACTION
# ============================================================

def _extract_gif_contact_sheet(
    source: Image.Image,
):

    try:

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

                source.load()

                frame = image_to_rgb(
                    source
                )

                frame = resize_for_vision(
                    frame
                )

                frames.append(
                    frame.copy()
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
            "%d total frame(s); using %d.",
            total_frames,
            len(frames),
        )

        return create_contact_sheet(
            frames
        )

    except Exception:

        logger.exception(
            "❌ GIF frame extraction failed."
        )

        return None


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image_for_vision(
    image_bytes: bytes,
):

    """
    Convert Telegram media into a clean JPEG-compatible
    image for Vision.

    Supported:

    JPEG
    PNG
    GIF
    WebP
    MP4/MOV
    WebM
    Telegram GIF-as-MP4
    """

    if not image_bytes:

        logger.error(
            "❌ prepare_image_for_vision received empty bytes."
        )

        return None

    logger.info(
        "🔬 Media signature: %s",
        get_media_signature(image_bytes),
    )

    logger.info(
        "📦 Media size: %d bytes",
        len(image_bytes),
    )

    try:

        # ----------------------------------------------------
        # VIDEO / TELEGRAM GIF
        # ----------------------------------------------------

        if _is_video_container(
            image_bytes
        ):

            logger.info(
                "🎞️ MP4/MOV/WebM container detected."
            )

            return _extract_video_contact_sheet(
                image_bytes
            )

        # ----------------------------------------------------
        # NORMAL IMAGE
        # ----------------------------------------------------

        logger.info(
            "🖼️ Attempting to decode media with Pillow..."
        )

        source = Image.open(
            BytesIO(image_bytes)
        )

        logger.info(
            "🖼️ Pillow detected format=%s mode=%s size=%s",
            source.format,
            source.mode,
            source.size,
        )

        source.load()

        # ----------------------------------------------------
        # ANIMATED GIF
        # ----------------------------------------------------

        if bool(
            getattr(
                source,
                "is_animated",
                False,
            )
        ):

            logger.info(
                "🎞️ Pillow detected animated media."
            )

            return _extract_gif_contact_sheet(
                source
            )

        # ----------------------------------------------------
        # STATIC IMAGE
        # ----------------------------------------------------

        rgb = image_to_rgb(
            source
        )

        rgb = resize_for_vision(
            rgb
        )

        logger.info(
            "✅ Static image prepared: %sx%s",
            rgb.width,
            rgb.height,
        )

        return rgb

    except Exception:

        logger.exception(
            "❌ IMAGE PREPARATION FAILED"
        )

        return None


# ============================================================
# VISION REQUEST
# ============================================================

async def describe_image_bytes(
    image_bytes: bytes,
):

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

        logger.info(
            "⏳ Vision cooldown: %.2fs",
            wait,
        )

        await asyncio.sleep(
            wait
        )

    _last_vision_call = time.time()

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    logger.info(
        "🖼️ Preparing original media for Vision..."
    )

    image = prepare_image_for_vision(
        image_bytes
    )

    if image is None:

        logger.error(
            "❌ Vision image preparation returned None."
        )

        return None

    # --------------------------------------------------------
    # JPEG ENCODE
    # --------------------------------------------------------

    try:

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=92,
            optimize=True,
        )

        jpeg_bytes = buffer.getvalue()

        logger.info(
            "✅ Vision JPEG prepared: %d bytes",
            len(jpeg_bytes),
        )

        encoded = base64.b64encode(
            jpeg_bytes
        ).decode("ascii")

        image_data_url = (
            "data:image/jpeg;base64,"
            + encoded
        )

    except Exception:

        logger.exception(
            "❌ Failed to encode Vision image."
        )

        return None

    # ========================================================
    # VISION PROMPT
    # ========================================================

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

If this is a contact sheet from a GIF/video:

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

    # ========================================================
    # VISION REQUEST
    # ========================================================

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

            # ------------------------------------------------
            # Use Responses API.
            #
            # This avoids the older chat-completions image
            # request path and is the current OpenAI API style.
            # ------------------------------------------------

            response = (
                openai_client.responses.create(
                    model=VISION_MODEL,

                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": prompt,
                                },
                                {
                                    "type": "input_image",
                                    "image_url": image_data_url,
                                    "detail": "high",
                                },
                            ],
                        }
                    ],
                )
            )

            description = getattr(
                response,
                "output_text",
                None,
            )

            if not description:

                logger.error(
                    "❌ Vision returned empty output."
                )

                logger.error(
                    "❌ Raw response: %r",
                    response,
                )

                return None

            description = description.strip()

            logger.info(
                "✅ Vision deconstruction complete (%d chars)",
                len(description),
            )

            logger.info(
                "🧩 Vision preview:\n%s",
                description[:2000],
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
                    "❌ API ERROR BODY: %r",
                    body,
                )

            response_obj = getattr(
                e,
                "response",
                None,
            )

            if response_obj:

                logger.error(
                    "❌ API RESPONSE: %r",
                    response_obj,
                )

            logger.error(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            # ------------------------------------------------
            # Retry rate limits / temporary errors.
            # ------------------------------------------------

            if (
                status_code in (
                    429,
                    500,
                    502,
                    503,
                    504,
                )
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
# IMAGE GENERATION
# ============================================================

async def generate_image_from_description(
    description: str,
):

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

        logger.info(
            "⏳ Image generation cooldown: %.2fs",
            wait,
        )

        await asyncio.sleep(
            wait
        )

    _last_generation_call = time.time()

    # ========================================================
    # GENERATION PROMPT
    # ========================================================

    generation_prompt = f"""
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

Make legitimate text as accurate and readable as possible.

Do not invent:

- scores
- odds
- dates
- teams
- players
- picks
- numbers

Only reproduce legitimate information that is actually
contained in the reconstruction specification.

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

    # ========================================================
    # GENERATION REQUEST
    # ========================================================

    max_retries = 2

    for attempt in range(
        max_retries
    ):

        try:

            logger.info(
                "🎨 OpenAI image generation request %d/%d",
                attempt + 1,
                max_retries,
            )

            logger.info(
                "🎨 Image model: %s",
                GENERATION_MODEL,
            )

            logger.info(
                "📝 Generation prompt length: %d characters",
                len(generation_prompt),
            )

            response = (
                openai_client.images.generate(
                    model=GENERATION_MODEL,
                    prompt=generation_prompt,
                    n=1,
                )
            )

            if not response:

                logger.error(
                    "❌ OpenAI returned no image response."
                )

                return None

            if not response.data:

                logger.error(
                    "❌ OpenAI returned no image data."
                )

                return None

            image_data = response.data[0]

            # ------------------------------------------------
            # BASE64
            # ------------------------------------------------

            b64_json = getattr(
                image_data,
                "b64_json",
                None,
            )

            if b64_json:

                try:

                    image_bytes = base64.b64decode(
                        b64_json
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

            # ------------------------------------------------
            # URL
            # ------------------------------------------------

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
                "❌ OpenAI response contained neither "
                "b64_json nor URL."
            )

            logger.error(
                "❌ Image response object: %r",
                image_data,
            )

            return None

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
                status_code,
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
                    "❌ API ERROR BODY: %r",
                    body,
                )

            response_obj = getattr(
                e,
                "response",
                None,
            )

            if response_obj:

                logger.error(
                    "❌ API RESPONSE: %r",
                    response_obj,
                )

            logger.error(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            if (
                status_code in (
                    429,
                    500,
                    502,
                    503,
                    504,
                )
                and attempt < max_retries - 1
            ):

                wait = 2 ** attempt

                logger.warning(
                    "⏳ Retrying image generation in %ds...",
                    wait,
                )

                await asyncio.sleep(
                    wait
                )

                continue

            return None

    return None


# ============================================================
# COMPLETE REGENERATION PIPELINE
# ============================================================

async def regenerate_image_from_bytes(
    image_bytes: bytes,
):

    """
    Complete pipeline:

        Telegram image/GIF/video
                ↓
        Detect media
                ↓
        Convert to JPEG/contact sheet
                ↓
        OpenAI Vision
                ↓
        Deconstruction
                ↓
        OpenAI Image Generation
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
        get_media_signature(image_bytes),
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
            "❌ No reconstruction specification was returned."
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
        "✅ STEP 2 COMPLETE"
    )

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
