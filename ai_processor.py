import os
import io
import re
import base64
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

openai_client = (
    OpenAI(api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY
    else None
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

if (
    NEW_MENTION
    and not NEW_MENTION.startswith("@")
):
    NEW_MENTION = "@" + NEW_MENTION


# ============================================================
# OPENAI CONFIGURATION
# ============================================================

OPENAI_IMAGE_MODEL = os.getenv(
    "OPENAI_IMAGE_MODEL",
    "gpt-image-2",
).strip()

OPENAI_IMAGE_QUALITY = os.getenv(
    "OPENAI_IMAGE_QUALITY",
    "high",
).strip()

OPENAI_INPUT_FIDELITY = os.getenv(
    "OPENAI_INPUT_FIDELITY",
    "high",
).strip()

# Number of edit attempts.
OPENAI_EDIT_MAX_ATTEMPTS = max(
    1,
    int(
        os.getenv(
            "OPENAI_EDIT_MAX_ATTEMPTS",
            "2",
        )
    ),
)


# ============================================================
# VERIFICATION
# ============================================================

WATERMARK_VERIFICATION_ENABLED = (
    os.getenv(
        "WATERMARK_VERIFICATION_ENABLED",
        "true",
    )
    .strip()
    .lower()
    not in {
        "0",
        "false",
        "no",
        "off",
    }
)

WATERMARK_VERIFY_MODEL = os.getenv(
    "WATERMARK_VERIFY_MODEL",
    "gpt-4o-mini",
).strip()


# ============================================================
# GIF SETTINGS
# ============================================================

GIF_SAMPLE_COUNT = max(
    3,
    int(
        os.getenv(
            "GIF_SAMPLE_COUNT",
            "12",
        )
    ),
)

# Prefer a frame in which the watermark is less visually
# prominent, but DO NOT use red alone as the final detector.
FRAME_SCORE_REDSUPPRESSION = True


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
    "🧠 Image model: %s",
    OPENAI_IMAGE_MODEL,
)

logger.info(
    "🎨 Quality: %s",
    OPENAI_IMAGE_QUALITY,
)

logger.info(
    "🔎 Verification: %s",
    (
        "enabled"
        if WATERMARK_VERIFICATION_ENABLED
        else "disabled"
    ),
)

if WATERMARK_VERIFICATION_ENABLED:
    logger.info(
        "🔎 Verification model: %s",
        WATERMARK_VERIFY_MODEL,
    )

logger.info(
    "🖼️ GIF workflow: representative still → OpenAI edit"
)

logger.info(
    "🚫 OpenCV watermark removal: DISABLED"
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
    ONLY:

    1. Remove *
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
        "👤 OLD: %r",
        OLD_MENTION,
    )

    logger.info(
        "👤 NEW: %r",
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
        data.startswith(b"GIF87a")
        or data.startswith(b"GIF89a")
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
# IMAGE HELPERS
# ============================================================

def pil_to_png_bytes(
    image: Image.Image,
) -> bytes:

    output = io.BytesIO()

    image.convert(
        "RGB"
    ).save(
        output,
        format="PNG",
    )

    return output.getvalue()


def decode_image(
    image_bytes: bytes,
) -> Image.Image | None:

    try:

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        image.load()

        return image.convert(
            "RGB"
        )

    except Exception:

        logger.exception(
            "❌ Could not decode image."
        )

        return None


# ============================================================
# REPRESENTATIVE FRAME SCORING
# ============================================================

def score_frame(
    image: Image.Image,
) -> float:
    """
    Lower score is preferred.

    This is only a frame-selection heuristic.

    It is intentionally conservative:
    red is one signal, but the model itself performs the actual
    watermark removal.
    """

    try:

        rgb = image.convert(
            "RGB"
        )

        # Downsample so a 1920x1080 GIF doesn't require scanning
        # millions of pixels for frame selection.
        rgb.thumbnail(
            (320, 320)
        )

        pixels = list(
            rgb.getdata()
        )

        if not pixels:
            return 0.0

        red_score = 0.0
        dark_red_score = 0.0

        for r, g, b in pixels:

            maximum = max(
                r,
                g,
                b,
            )

            minimum = min(
                r,
                g,
                b,
            )

            saturation_proxy = (
                maximum
                - minimum
            )

            dominance = (
                r
                - max(
                    g,
                    b,
                )
            )

            if dominance > 18:

                red_score += (
                    dominance
                    / 255.0
                )

            # Faded red can have much lower saturation.
            if (
                dominance > 8
                and saturation_proxy > 10
            ):

                dark_red_score += (
                    dominance
                    / 255.0
                )

        pixel_count = len(
            pixels
        )

        return (
            red_score / pixel_count
            +
            0.5
            * (
                dark_red_score
                / pixel_count
            )
        )

    except Exception:

        return 0.0


# ============================================================
# CHOOSE FRAME
# ============================================================

def choose_best_frame(
    frames: list[tuple[int, Image.Image]],
):
    """
    Pick the best representative frame.

    We choose among the sampled frames rather than blindly
    taking frame 1.

    Red detection is only used to help avoid the strongest
    watermark state; OpenAI still performs the actual cleanup.
    """

    if not frames:
        return None

    if len(frames) == 1:
        return frames[0][1]

    scored = []

    for index, frame in frames:

        score = score_frame(
            frame
        )

        scored.append(
            (
                score,
                index,
                frame,
            )
        )

    scored.sort(
        key=lambda item: item[0]
    )

    best_score, best_index, best_frame = (
        scored[0]
    )

    logger.info(
        "🎯 Representative frame selected: %d "
        "(score %.6f)",
        best_index,
        best_score,
    )

    return best_frame


# ============================================================
# GIF FRAME EXTRACTION
# ============================================================

def extract_best_gif_frame(
    image_bytes: bytes,
):
    """
    Extract sampled frames from a GIF and choose one
    representative still.
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

        sample_count = min(
            GIF_SAMPLE_COUNT,
            total_frames,
        )

        if sample_count == 1:

            indexes = [0]

        else:

            indexes = [
                round(
                    i
                    * (
                        total_frames - 1
                    )
                    / (
                        sample_count - 1
                    )
                )
                for i in range(
                    sample_count
                )
            ]

        frames = []

        for index in indexes:

            try:

                source.seek(
                    index
                )

                frame = (
                    source
                    .convert("RGB")
                    .copy()
                )

                frames.append(
                    (
                        index,
                        frame,
                    )
                )

            except Exception:

                logger.exception(
                    "⚠️ Could not read GIF frame %s.",
                    index,
                )

        if not frames:
            return None

        return choose_best_frame(
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
    Telegram commonly delivers GIFs as MP4.

    We extract a small number of representative frames and
    choose one to send to OpenAI.
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

            # Sample roughly every quarter second and cap the
            # number of frames.
            command = [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vf",
                "fps=4",
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
                    "❌ FFmpeg failed extracting MP4 frames:"
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
                    "❌ No representative MP4 frames found."
                )

                return None

            frames = []

            for index, path in enumerate(
                frame_paths
            ):

                try:

                    with Image.open(
                        path
                    ) as image:

                        frames.append(
                            (
                                index,
                                image.convert(
                                    "RGB"
                                ).copy(),
                            )
                        )

                except Exception:

                    logger.exception(
                        "⚠️ Could not read %s",
                        path,
                    )

            if not frames:
                return None

            logger.info(
                "🎞️ Extracted %d MP4 sample frame(s).",
                len(frames),
            )

            return choose_best_frame(
                frames
            )

    except Exception:

        logger.exception(
            "❌ MP4 frame extraction failed."
        )

        return None


# ============================================================
# PREPARE STILL IMAGE
# ============================================================

def prepare_still_image(
    image_bytes: bytes,
):
    """
    Convert incoming media into one PNG still.

    GIF/MP4:
        sample frames → choose representative frame

    Image:
        preserve image content → PNG
    """

    if not image_bytes:
        return None

    if is_gif(
        image_bytes
    ):

        logger.info(
            "🎞️ Actual GIF detected."
        )

        frame = extract_best_gif_frame(
            image_bytes
        )

        if frame is None:
            return None

        return pil_to_png_bytes(
            frame
        )

    if is_video_container(
        image_bytes
    ):

        logger.info(
            "🎞️ MP4/video detected."
        )

        frame = extract_best_mp4_frame(
            image_bytes
        )

        if frame is None:
            return None

        return pil_to_png_bytes(
            frame
        )

    image = decode_image(
        image_bytes
    )

    if image is None:
        return None

    return pil_to_png_bytes(
        image
    )


# ============================================================
# OPENAI EDIT PROMPT
# ============================================================

WATERMARK_EDIT_PROMPT = """
Perform a precise image cleanup.

REMOVE ONLY:
1. every visible, faint, translucent, partial, or stylized
   "@cappersfree" / "cappersfree" watermark
2. every CF graphic/logo associated with that watermark

The goal is NOT to redesign the image.

Reconstruct the area underneath the removed watermark naturally
so the final result looks like a clean original graphic.

PRESERVE EXACTLY AS MUCH AS POSSIBLE:
- legitimate text
- scores
- odds
- numbers
- dates
- team names
- player names
- faces
- people
- uniforms
- icons unrelated to CF
- layout
- composition
- spacing
- panels
- borders
- colors
- gradients
- shadows
- typography
- background details
- aspect ratio

IMPORTANT:
- Do not alter legitimate information.
- Do not rewrite or retype legitimate text.
- Do not invent new information.
- Do not add a new logo.
- Do not add a new username.
- Do not replace the watermark with branding.
- Do not crop the image.
- Do not redesign the graphic.
- Do not intentionally blur the watermark area.
- Reconstruct the missing background/content naturally.

The ONLY intended modification is removal of the CF graphic/logo
and @cappersfree watermark.
"""


WATERMARK_RETRY_PROMPT = """
The previous edit was rejected because the watermark was still
visible or the result was uncertain.

Perform another careful cleanup.

Search the ENTIRE image again for:
- red @cappersfree
- faint red @cappersfree
- translucent @cappersfree
- partial cappersfree text
- CF graphic
- faded CF graphic
- pulsing-logo appearance captured in this still

Remove all instances.

Do not leave a faint ghost of the watermark.

At the same time, preserve all legitimate text, numbers,
players, faces, layout, colors, and sports information.
Do not redesign the image.
"""


# ============================================================
# VERIFICATION PROMPT
# ============================================================

WATERMARK_VERIFICATION_PROMPT = """
You are the final quality-control checker for a watermark-removal
pipeline.

Inspect the IMAGE itself.

The target watermark is:
- a CF graphic/logo
- "@cappersfree"
- "cappersfree"
- faint, translucent, faded, partial, or stylized versions
of those marks

Do not treat legitimate sports content as a watermark.

Reply with EXACTLY one of:

CLEAN

or

NOT_CLEAN

Reply NOT_CLEAN if:
- any CF logo remains
- any @cappersfree remains
- any faint/ghosted watermark remains
- any partial watermark remains
- you are uncertain

Reply CLEAN only when the watermark is genuinely absent.
"""


# ============================================================
# IMAGE VALIDATION
# ============================================================

def is_valid_image_bytes(
    image_bytes: bytes,
) -> bool:

    if not image_bytes:
        return False

    try:

        with Image.open(
            io.BytesIO(
                image_bytes
            )
        ) as image:

            image.verify()

        with Image.open(
            io.BytesIO(
                image_bytes
            )
        ) as image:

            width, height = (
                image.size
            )

        return (
            width > 0
            and height > 0
        )

    except Exception:

        logger.exception(
            "❌ Returned image is invalid."
        )

        return False


# ============================================================
# VERIFY CLEAN RESULT
# ============================================================

def verify_watermark_is_removed(
    image_bytes: bytes,
) -> bool:
    """
    Independent vision check.

    Fail closed:
        if verification fails → do not publish.
    """

    if not WATERMARK_VERIFICATION_ENABLED:

        logger.warning(
            "⚠️ Watermark verification disabled."
        )

        return True

    if not openai_client:

        logger.error(
            "❌ OpenAI client unavailable."
        )

        return False

    if not is_valid_image_bytes(
        image_bytes
    ):

        return False

    try:

        encoded = base64.b64encode(
            image_bytes
        ).decode(
            "ascii"
        )

        response = (
            openai_client
            .responses
            .create(
                model=WATERMARK_VERIFY_MODEL,
                instructions=(
                    WATERMARK_VERIFICATION_PROMPT
                ),
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Check this edited image "
                                    "before publication."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": (
                                    "data:image/png;base64,"
                                    f"{encoded}"
                                ),
                                "detail": "high",
                            },
                        ],
                    }
                ],
                max_output_tokens=10,
            )
        )

        decision = (
            getattr(
                response,
                "output_text",
                "",
            )
            or ""
        ).strip()

        logger.info(
            "🔎 Verification response: %r",
            decision,
        )

        if re.fullmatch(
            r"CLEAN[.!]?",
            decision,
            flags=re.IGNORECASE,
        ):

            logger.info(
                "✅ Watermark verification passed."
            )

            return True

        logger.warning(
            "❌ Watermark verification rejected result."
        )

        return False

    except Exception:

        logger.exception(
            "❌ Watermark verification failed."
        )

        return False


# ============================================================
# OPENAI IMAGE EDIT
# ============================================================

def edit_image_with_openai(
    image_bytes: bytes,
    prompt: str,
):

    if not openai_client:

        logger.error(
            "❌ OpenAI client is not available."
        )

        return None

    image_file = io.BytesIO(
        image_bytes
    )

    image_file.name = (
        "source.png"
    )

    logger.info(
        "🤖 Calling OpenAI image-edit endpoint..."
    )

    logger.info(
        "🧠 Model: %s",
        OPENAI_IMAGE_MODEL,
    )

    response = (
        openai_client
        .images
        .edit(
            model=OPENAI_IMAGE_MODEL,
            image=image_file,
            prompt=prompt,
            size="auto",
            quality=OPENAI_IMAGE_QUALITY,
            input_fidelity=OPENAI_INPUT_FIDELITY,
            output_format="png",
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

    result = response.data[0]

    b64_json = getattr(
        result,
        "b64_json",
        None,
    )

    if not b64_json:

        logger.error(
            "❌ OpenAI returned no b64_json."
        )

        return None

    try:

        output_bytes = base64.b64decode(
            b64_json
        )

    except Exception:

        logger.exception(
            "❌ Could not decode OpenAI image."
        )

        return None

    logger.info(
        "✅ OpenAI returned %d bytes.",
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
    MAIN IMAGE PIPELINE:

        Telegram image/GIF
               ↓
        choose representative still
               ↓
        OpenAI image edit
               ↓
        independent verification
               ↓
        optional second edit
               ↓
        CLEAN PNG

    GIFs become STILL IMAGES by design.
    """

    if not image_bytes:

        logger.error(
            "❌ Empty media."
        )

        return None

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    logger.info(
        "🧹 OPENAI WATERMARK PIPELINE START"
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

        # ====================================================
        # STEP 1
        # ====================================================

        logger.info(
            "1️⃣ Preparing representative still image..."
        )

        still_bytes = await asyncio.to_thread(
            prepare_still_image,
            image_bytes,
        )

        if not still_bytes:

            logger.error(
                "❌ Could not prepare representative still."
            )

            return None

        logger.info(
            "✅ Still prepared: %d bytes",
            len(still_bytes),
        )

        # ====================================================
        # STEP 2
        # ====================================================

        for attempt in range(
            1,
            OPENAI_EDIT_MAX_ATTEMPTS + 1,
        ):

            logger.info(
                "2️⃣ OpenAI edit attempt %d/%d",
                attempt,
                OPENAI_EDIT_MAX_ATTEMPTS,
            )

            prompt = (
                WATERMARK_EDIT_PROMPT
            )

            if attempt > 1:

                prompt += (
                    "\n\n"
                    + WATERMARK_RETRY_PROMPT
                )

            edited_bytes = await asyncio.to_thread(
                edit_image_with_openai,
                still_bytes,
                prompt,
            )

            if not edited_bytes:

                logger.error(
                    "❌ OpenAI edit failed on attempt %d.",
                    attempt,
                )

                continue

            if not is_valid_image_bytes(
                edited_bytes
            ):

                logger.error(
                    "❌ Invalid image returned on attempt %d.",
                    attempt,
                )

                continue

            # =================================================
            # STEP 3 — VERIFY
            # =================================================

            verified = await asyncio.to_thread(
                verify_watermark_is_removed,
                edited_bytes,
            )

            if not verified:

                logger.warning(
                    "⚠️ Attempt %d failed verification.",
                    attempt,
                )

                continue

            logger.info(
                "✅ VERIFIED CLEAN IMAGE"
            )

            return BufferedInputFile(
                edited_bytes,
                filename="cleaned.png",
            )

        logger.error(
            "❌ No verified clean image was produced."
        )

        logger.error(
            "🚫 Original media will NOT be published."
        )

        return None

    except Exception as e:

        logger.exception(
            "❌ OPENAI WATERMARK PIPELINE FAILED"
        )

        logger.error(
            "❌ Exception: %s",
            type(e).__name__,
        )

        logger.error(
            "❌ Message: %s",
            str(e),
        )

        logger.error(
            "❌ Status: %s",
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
            "🧹 OPENAI WATERMARK PIPELINE END"
        )

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
