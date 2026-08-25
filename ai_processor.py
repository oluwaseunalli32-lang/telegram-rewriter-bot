import os
import io
import re
import asyncio
import logging
import tempfile
import subprocess

from pathlib import Path

from PIL import Image, ImageSequence
from openai import OpenAI
from aiogram.types import BufferedInputFile


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("ai_processor")


# ============================================================
# ENVIRONMENT
# ============================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
).strip()

if not OPENAI_API_KEY:
    logger.error(
        "❌ OPENAI_API_KEY is missing."
    )

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
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
# OPENAI IMAGE SETTINGS
# ============================================================

OPENAI_IMAGE_MODEL = os.getenv(
    "OPENAI_IMAGE_MODEL",
    "gpt-image-2",
).strip()

OPENAI_IMAGE_QUALITY = os.getenv(
    "OPENAI_IMAGE_QUALITY",
    "high",
).strip()

# We use high input fidelity so the model tries to preserve
# the source image's visual details.
OPENAI_INPUT_FIDELITY = "high"


# ============================================================
# GIF SETTINGS
# ============================================================

GIF_SAMPLE_COUNT = int(
    os.getenv(
        "GIF_SAMPLE_COUNT",
        "12",
    )
)


# ============================================================
# STARTUP
# ============================================================

logger.info(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

logger.info(
    "🤖 OPENAI WATERMARK EDITOR STARTED"
)

logger.info(
    "🧠 Model: %s",
    OPENAI_IMAGE_MODEL,
)

logger.info(
    "🎨 Quality: %s",
    OPENAI_IMAGE_QUALITY,
)

logger.info(
    "🖼️ Workflow: image/GIF → still frame → OpenAI edit"
)

logger.info(
    "🚫 Full-image reconstruction pipeline: DISABLED"
)

logger.info(
    "🚫 OpenCV watermark inpainting: DISABLED"
)

logger.info(
    "🚫 Replicate: DISABLED"
)

logger.info(
    "📝 Caption AI rewriting: DISABLED"
)

logger.info(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)


# ============================================================
# CAPTION PROCESSING
# ============================================================

def replace_username(
    text: str,
) -> str:

    """
    Existing caption behavior:

        1. Remove every *
        2. Replace @cappersfree with NEW_MENTION

    Nothing else.
    """

    if not text:
        return text

    result = text.replace(
        "*",
        "",
    )

    if NEW_MENTION:

        result = re.sub(
            re.escape(
                OLD_MENTION
            ),
            NEW_MENTION,
            result,
            flags=re.IGNORECASE,
        )

    else:

        logger.warning(
            "⚠️ NEW_MENTION is empty."
        )

    return result


async def rewrite_text(
    original_text: str,
) -> str:

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
# MEDIA DETECTION
# ============================================================

def is_gif(
    data: bytes,
) -> bool:

    return (
        data.startswith(
            b"GIF87a"
        )
        or
        data.startswith(
            b"GIF89a"
        )
    )


def is_video_container(
    data: bytes,
) -> bool:

    if (
        len(data) >= 12
        and data[4:8] == b"ftyp"
    ):
        return True

    if data.startswith(
        b"\x1a\x45\xdf\xa3"
    ):
        return True

    return False


# ============================================================
# IMAGE CONVERSION
# ============================================================

def pil_to_png_bytes(
    image: Image.Image,
) -> bytes:

    output = io.BytesIO()

    # RGB is safest for the image-edit API.
    image.convert(
        "RGB"
    ).save(
        output,
        format="PNG",
    )

    return output.getvalue()


# ============================================================
# GIF FRAME SELECTION
# ============================================================

def _frame_red_score(
    image: Image.Image,
) -> float:

    """
    Lightweight heuristic used only to choose a representative
    GIF frame.

    We are NOT using this as a watermark remover.

    The goal is simply to avoid selecting an extremely strong
    red-watermark frame when another frame is available.
    """

    try:

        rgb = image.convert(
            "RGB"
        )

        pixels = list(
            rgb.getdata()
        )

        if not pixels:
            return 0.0

        red_like = 0

        for r, g, b in pixels:

            if (
                r > g * 1.20
                and r > b * 1.20
                and r > 80
            ):

                red_like += 1

        return (
            red_like
            / len(pixels)
        )

    except Exception:

        return 0.0


def choose_best_gif_frame(
    frames,
):
    """
    Choose a representative still frame.

    We sample the GIF and prefer a frame with relatively little
    obvious red-overlay activity.

    IMPORTANT:
    OpenAI is still responsible for actually removing the
    watermark. This local score is only frame selection.
    """

    if not frames:
        return None

    if len(frames) == 1:
        return frames[0]

    candidates = []

    for index, frame in enumerate(
        frames
    ):

        score = _frame_red_score(
            frame
        )

        candidates.append(
            (
                score,
                index,
                frame,
            )
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    best = candidates[0]

    logger.info(
        "🎯 Selected GIF frame %d/%d "
        "(red-overlay score %.6f)",
        best[1] + 1,
        len(frames),
        best[0],
    )

    return best[2]


def extract_best_gif_frame(
    image_bytes: bytes,
):
    """
    Extract a representative still from an actual GIF.
    """

    try:

        source = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        total_frames = int(
            getattr(
                source,
                "n_frames",
                1,
            )
        )

        logger.info(
            "🎞️ GIF contains %d frame(s).",
            total_frames,
        )

        if total_frames <= GIF_SAMPLE_COUNT:

            indexes = list(
                range(
                    total_frames
                )
            )

        else:

            indexes = [
                round(
                    i
                    * (
                        total_frames - 1
                    )
                    / (
                        GIF_SAMPLE_COUNT - 1
                    )
                )
                for i in range(
                    GIF_SAMPLE_COUNT
                )
            ]

        frames = []

        for index in indexes:

            try:

                source.seek(
                    index
                )

                frame = source.convert(
                    "RGB"
                ).copy()

                frames.append(
                    frame
                )

            except Exception:

                logger.exception(
                    "⚠️ Could not read GIF frame %s.",
                    index,
                )

        if not frames:
            return None

        return choose_best_gif_frame(
            frames
        )

    except Exception:

        logger.exception(
            "❌ GIF frame extraction failed."
        )

        return None


# ============================================================
# MP4 FRAME EXTRACTION
# ============================================================

def extract_best_mp4_frame(
    video_bytes: bytes,
):
    """
    Telegram commonly sends animated GIFs as MP4.

    Extract a representative still frame locally.
    """

    try:

        from imageio_ffmpeg import (
            get_ffmpeg_exe,
        )

        ffmpeg = get_ffmpeg_exe()

    except Exception:

        logger.exception(
            "❌ imageio-ffmpeg is unavailable."
        )

        return None

    try:

        with tempfile.TemporaryDirectory() as temp_dir:

            temp = Path(
                temp_dir
            )

            input_path = (
                temp
                / "input.mp4"
            )

            frames_dir = (
                temp
                / "frames"
            )

            frames_dir.mkdir()

            input_path.write_bytes(
                video_bytes
            )

            output_pattern = str(
                frames_dir
                / "frame_%04d.png"
            )

            # Extract a limited number of evenly distributed
            # frames. The actual frame count is unknown, so we
            # first ask FFmpeg to sample at a low rate.
            command = [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vf",
                (
                    f"fps="
                    f"min(12,"
                    f"1000)"
                ),
                "-frames:v",
                str(
                    GIF_SAMPLE_COUNT
                ),
                output_pattern,
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )

            if result.returncode != 0:

                logger.error(
                    "❌ FFmpeg frame extraction failed:"
                )

                logger.error(
                    result.stderr.decode(
                        "utf-8",
                        errors="replace",
                    )[-3000:]
                )

                return None

            frame_paths = sorted(
                frames_dir.glob(
                    "frame_*.png"
                )
            )

            if not frame_paths:

                logger.error(
                    "❌ FFmpeg produced no frames."
                )

                return None

            frames = []

            for path in frame_paths:

                try:

                    with Image.open(
                        path
                    ) as frame:

                        frames.append(
                            frame.convert(
                                "RGB"
                            ).copy()
                        )

                except Exception:

                    logger.exception(
                        "⚠️ Could not read %s",
                        path,
                    )

            if not frames:

                return None

            logger.info(
                "🎞️ Extracted %d representative MP4 frames.",
                len(frames),
            )

            return choose_best_gif_frame(
                frames
            )

    except Exception:

        logger.exception(
            "❌ MP4 frame extraction failed."
        )

        return None


# ============================================================
# PREPARE ORIGINAL MEDIA AS STILL IMAGE
# ============================================================

def prepare_still_image(
    image_bytes: bytes,
):
    """
    Convert any supported input into one PNG still:

        PNG/JPG/WEBP → itself converted to PNG
        GIF          → representative frame
        MP4          → representative frame
    """

    if not image_bytes:
        return None

    # --------------------------------------------------------
    # Actual GIF.
    # --------------------------------------------------------

    if is_gif(
        image_bytes
    ):

        logger.info(
            "🎞️ Actual GIF detected. "
            "Selecting representative still..."
        )

        frame = extract_best_gif_frame(
            image_bytes
        )

        if frame is None:
            return None

        return pil_to_png_bytes(
            frame
        )

    # --------------------------------------------------------
    # MP4 / video.
    # --------------------------------------------------------

    if is_video_container(
        image_bytes
    ):

        logger.info(
            "🎞️ MP4/GIF video detected. "
            "Selecting representative still..."
        )

        frame = extract_best_mp4_frame(
            image_bytes
        )

        if frame is None:
            return None

        return pil_to_png_bytes(
            frame
        )

    # --------------------------------------------------------
    # Normal image.
    # --------------------------------------------------------

    try:

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        image.load()

        return pil_to_png_bytes(
            image
        )

    except Exception:

        logger.exception(
            "❌ Could not convert image to PNG."
        )

        return None


# ============================================================
# OPENAI EDIT PROMPT
# ============================================================

WATERMARK_EDIT_PROMPT = """
Edit this image to remove ONLY the CF graphic/logo and any
visible or faint "@cappersfree" watermark.

This is an image cleanup/edit, NOT a redesign and NOT a new
sports graphic.

Preserve everything else from the source image as faithfully
as possible:

- all legitimate text
- all numbers
- scores
- odds
- dates
- team/player names
- faces and people
- icons unrelated to the watermark
- colors
- typography
- layout
- spacing
- borders
- panels
- background
- proportions
- overall visual design

Where the CF logo or @cappersfree watermark was present,
reconstruct the underlying background/content naturally so
there is no visible blur, patch, smudge, or blank area.

Do NOT replace the watermark with another logo.
Do NOT add any new branding.
Do NOT add a username.
Do NOT redesign the image.
Do NOT change legitimate text.
Do NOT change numbers or sports information.

The only requested edit is:
REMOVE THE CF GRAPHIC/LOGO AND @CAPPERSFREE WATERMARK.
"""


# ============================================================
# OPENAI IMAGE EDIT
# ============================================================

def edit_image_with_openai(
    image_bytes: bytes,
):
    """
    Send one prepared still image to GPT-Image-2.

    Uses the image-edit endpoint rather than image generation.
    """

    if not OPENAI_API_KEY:

        logger.error(
            "❌ OPENAI_API_KEY is missing."
        )

        return None

    # --------------------------------------------------------
    # The current OpenAI image-edit API expects a supported
    # image file. PNG is safest here.
    # --------------------------------------------------------

    image_file = io.BytesIO(
        image_bytes
    )

    # Some OpenAI Python SDK versions inspect the file name
    # when building multipart form data.
    image_file.name = (
        "source.png"
    )

    logger.info(
        "🤖 Sending still image to OpenAI image-edit API..."
    )

    logger.info(
        "🧠 Model: %s",
        OPENAI_IMAGE_MODEL,
    )

    logger.info(
        "🎨 Quality: %s",
        OPENAI_IMAGE_QUALITY,
    )

    response = openai_client.images.edit(
        model=OPENAI_IMAGE_MODEL,
        image=image_file,
        prompt=WATERMARK_EDIT_PROMPT,
        size="auto",
        quality=OPENAI_IMAGE_QUALITY,
        input_fidelity=OPENAI_INPUT_FIDELITY,
        output_format="png",
        moderation="auto",
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

    result = response.data[0]

    b64_json = getattr(
        result,
        "b64_json",
        None,
    )

    if not b64_json:

        logger.error(
            "❌ OpenAI image-edit response did not contain b64_json."
        )

        return None

    import base64

    output_bytes = base64.b64decode(
        b64_json
    )

    logger.info(
        "✅ OpenAI returned edited image: %d bytes",
        len(output_bytes),
    )

    return output_bytes


# ============================================================
# PUBLIC MEDIA FUNCTION
# ============================================================

async def remove_watermarks_from_bytes(
    image_bytes: bytes,
    filename: str = "",
    mime_type: str = "",
):
    """
    Main media-cleaning function.

    IMPORTANT:

    Every GIF/MP4 is first converted to ONE still image.

    Then that still is sent to OpenAI's image-edit endpoint.

    Result:
        clean PNG BufferedInputFile

    No Replicate.
    No OpenCV watermark removal.
    No full-image reconstruction prompt.
    """

    if not image_bytes:

        logger.error(
            "❌ Empty media bytes."
        )

        return None

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    logger.info(
        "🧹 OPENAI WATERMARK EDIT PIPELINE START"
    )

    logger.info(
        "📦 Original bytes: %d",
        len(image_bytes),
    )

    logger.info(
        "📄 Filename: %s",
        filename,
    )

    logger.info(
        "📄 MIME: %s",
        mime_type,
    )

    try:

        # ----------------------------------------------------
        # STEP 1: Convert media to one still image.
        # ----------------------------------------------------

        logger.info(
            "1️⃣ STEP 1/2 — Preparing still image..."
        )

        still_bytes = await asyncio.to_thread(
            prepare_still_image,
            image_bytes,
        )

        if not still_bytes:

            logger.error(
                "❌ Could not prepare still image."
            )

            return None

        logger.info(
            "✅ Still image prepared: %d bytes",
            len(still_bytes),
        )

        # ----------------------------------------------------
        # STEP 2: OpenAI edit.
        # ----------------------------------------------------

        logger.info(
            "2️⃣ STEP 2/2 — OpenAI watermark removal..."
        )

        edited_bytes = await asyncio.to_thread(
            edit_image_with_openai,
            still_bytes,
        )

        if not edited_bytes:

            logger.error(
                "❌ OpenAI image edit failed."
            )

            return None

        logger.info(
            "✅ CLEAN STILL IMAGE CREATED"
        )

        logger.info(
            "✅ OPENAI WATERMARK EDIT COMPLETE"
        )

        return BufferedInputFile(
            edited_bytes,
            filename="cleaned.png",
        )

    except Exception as e:

        logger.exception(
            "❌ OPENAI WATERMARK EDIT FAILED"
        )

        logger.error(
            "❌ Exception type: %s",
            type(e).__name__,
        )

        logger.error(
            "❌ Error: %s",
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

        return None

    finally:

        logger.info(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )


# ============================================================
# OLD FUNCTION NAME COMPATIBILITY
# ============================================================

async def regenerate_image_from_bytes(
    image_bytes: bytes,
):

    return await remove_watermarks_from_bytes(
        image_bytes
    )
