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
from openai import OpenAI, RateLimitError
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
    OpenAI(
        api_key=OPENAI_API_KEY,
        max_retries=0,
    )
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
# GIF CONFIGURATION
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
    "🎞️ GIF → representative still → OpenAI edit"
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
):

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
# GIF FRAME SCORING
# ============================================================

def score_frame(
    image: Image.Image,
) -> float:

    """
    Frame selection only.

    This does NOT remove the watermark.
    """

    try:

        rgb = image.convert(
            "RGB"
        )

        rgb.thumbnail(
            (320, 320)
        )

        pixels = list(
            rgb.getdata()
        )

        if not pixels:
            return 0.0

        score = 0.0

        for r, g, b in pixels:

            dominance = (
                r
                - max(
                    g,
                    b,
                )
            )

            if dominance > 18:

                score += (
                    dominance
                    / 255.0
                )

        return (
            score
            / len(pixels)
        )

    except Exception:

        return 0.0


# ============================================================
# CHOOSE GIF FRAME
# ============================================================

def choose_best_frame(
    frames,
):

    if not frames:
        return None

    if len(frames) == 1:
        return frames[0][1]

    scored = []

    for index, frame in frames:

        scored.append(
            (
                score_frame(frame),
                index,
                frame,
            )
        )

    scored.sort(
        key=lambda item: item[0]
    )

    score, index, frame = (
        scored[0]
    )

    logger.info(
        "🎯 Representative frame selected: %d "
        "(score %.6f)",
        index,
        score,
    )

    return frame


# ============================================================
# ACTUAL GIF
# ============================================================

def extract_best_gif_frame(
    image_bytes: bytes,
):

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

        if sample_count <= 1:

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
# MP4 / TELEGRAM GIF
# ============================================================

def extract_best_mp4_frame(
    video_bytes: bytes,
):

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

            # IMPORTANT:
            #
            # Do not use:
            #     fps=min(12,1000)
            #
            # That was the source of the FFmpeg
            # "No such filter: 1000)" error.
            #
            # We simply sample at 4 fps and cap the
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
# PREPARE REPRESENTATIVE STILL
# ============================================================

def prepare_still_image(
    image_bytes: bytes,
):

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
# OPENAI PROMPTS
# ============================================================

WATERMARK_EDIT_PROMPT = """
Perform a precise image cleanup.

REMOVE ONLY:
1. every visible, faint, translucent, partial, or stylized
   "@cappersfree" / "cappersfree" watermark
2. every CF graphic/logo associated with that watermark

This is NOT a redesign.

Reconstruct the background/content underneath the removed
watermark naturally.

Preserve everything else as faithfully as possible:
- legitimate text
- scores
- odds
- numbers
- dates
- teams
- players
- faces
- people
- uniforms
- unrelated icons
- typography
- colors
- gradients
- shadows
- panels
- borders
- spacing
- composition
- layout
- aspect ratio

Do NOT:
- crop
- redesign
- rewrite legitimate text
- change numbers
- change scores
- change odds
- change people
- add logos
- add usernames
- add branding
- replace the watermark with another graphic
- blur the watermark area

The ONLY intended modification is removal of the
CF graphic/logo and @cappersfree watermark.
"""


WATERMARK_RETRY_PROMPT = """
The previous result was rejected because the CF or
@cappersfree watermark may still be visible.

Inspect the entire image again and remove:
- strong @cappersfree
- faint @cappersfree
- translucent @cappersfree
- partial cappersfree text
- CF logos
- faint CF graphics
- ghosted remnants of the watermark

Reconstruct the underlying background naturally.

Do not alter legitimate text, numbers, scores, odds,
players, faces, colors, typography, composition, or layout.
Do not redesign the image.
"""


WATERMARK_VERIFICATION_PROMPT = """
Inspect this edited image for the specific watermark we are
trying to remove.

Target:
- CF graphic/logo
- @cappersfree
- cappersfree
- faint versions
- translucent versions
- partial versions
- ghosted remnants

Do not classify legitimate sports graphics as the watermark.

Reply with EXACTLY one of:

CLEAN

or

NOT_CLEAN

Reply NOT_CLEAN if any target watermark remains or if you
are uncertain.

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
# OPENAI IMAGE EDIT
# ============================================================

def edit_image_with_openai(
    image_bytes: bytes,
    prompt: str,
):

    if not openai_client:

        logger.error(
            "❌ OpenAI client is unavailable."
        )

        return None

    image_file = io.BytesIO(
        image_bytes
    )

    image_file.name = (
        "source.png"
    )

    try:

        logger.info(
            "🤖 Calling OpenAI image-edit endpoint..."
        )

        logger.info(
            "🧠 Model: %s",
            OPENAI_IMAGE_MODEL,
        )

        logger.info(
            "🎨 Quality: %s",
            OPENAI_IMAGE_QUALITY,
        )

        # IMPORTANT:
        # Do NOT add moderation=...
        # The images.edit endpoint does not accept that
        # argument in the SDK version being used.
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

    except RateLimitError as exc:

        body = getattr(
            exc,
            "body",
            None,
        )

        body_text = str(
            body
            if body
            else exc
        ).lower()

        # ----------------------------------------------------
        # ZERO API CREDITS / QUOTA
        # ----------------------------------------------------

        if (
            "insufficient_quota"
            in body_text
            or
            "credit_balance_exhausted"
            in body_text
            or
            "no credits remaining"
            in body_text
        ):

            logger.error(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            logger.error(
                "❌ OPENAI API CREDITS EXHAUSTED"
            )

            logger.error(
                "❌ Add API credits to the OpenAI "
                "organization/project used by OPENAI_API_KEY."
            )

            logger.error(
                "❌ This is NOT an image-processing error."
            )

            logger.error(
                "❌ The request cannot succeed until "
                "API billing/credits are available."
            )

            logger.error(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            # Special marker understood by the caller.
            return "__OPENAI_CREDITS_EXHAUSTED__"

        # ----------------------------------------------------
        # Other rate limit.
        # ----------------------------------------------------

        logger.error(
            "❌ OpenAI rate limit error: %s",
            exc,
        )

        return None

    except Exception as exc:

        logger.exception(
            "❌ OpenAI image edit failed."
        )

        logger.error(
            "❌ Exception type: %s",
            type(exc).__name__,
        )

        logger.error(
            "❌ Error: %s",
            str(exc),
        )

        body = getattr(
            exc,
            "body",
            None,
        )

        if body:

            logger.error(
                "❌ API BODY: %s",
                body,
            )

        return None

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

        output_bytes = (
            base64.b64decode(
                b64_json
            )
        )

    except Exception:

        logger.exception(
            "❌ Failed decoding OpenAI image."
        )

        return None

    logger.info(
        "✅ OpenAI returned %d bytes.",
        len(output_bytes),
    )

    return output_bytes


# ============================================================
# WATERMARK VERIFICATION
# ============================================================

def verify_watermark_is_removed(
    image_bytes: bytes,
):

    if not WATERMARK_VERIFICATION_ENABLED:

        logger.warning(
            "⚠️ Watermark verification is disabled."
        )

        return True

    if not openai_client:

        return False

    if not is_valid_image_bytes(
        image_bytes
    ):

        return False

    try:

        encoded = (
            base64.b64encode(
                image_bytes
            )
            .decode(
                "ascii"
            )
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
                                    "Perform the final "
                                    "watermark quality check."
                                ),
                            },
                            {
                                "type": "input_image",
                                "image_url": (
                                    "data:image/png;base64,"
                                    + encoded
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
# MAIN PUBLIC PIPELINE
# ============================================================

async def remove_watermarks_from_bytes(
    image_bytes: bytes,
    filename: str = "",
    mime_type: str = "",
):

    if not image_bytes:

        logger.error(
            "❌ Empty media bytes."
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
        # STEP 1 — STILL
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
                "❌ Could not prepare still image."
            )

            return None

        logger.info(
            "✅ Still prepared: %d bytes",
            len(still_bytes),
        )

        # ====================================================
        # STEP 2 — EDIT
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

            edited_bytes = (
                await asyncio.to_thread(
                    edit_image_with_openai,
                    still_bytes,
                    prompt,
                )
            )

            # ------------------------------------------------
            # NO CREDITS.
            #
            # Do not make another attempt. There is nothing
            # the code can do until the account has credits.
            # ------------------------------------------------

            if (
                edited_bytes
                == "__OPENAI_CREDITS_EXHAUSTED__"
            ):

                logger.error(
                    "🚫 Stopping immediately because "
                    "OpenAI API credits are exhausted."
                )

                return None

            if not edited_bytes:

                logger.error(
                    "❌ OpenAI edit returned no image."
                )

                continue

            if not is_valid_image_bytes(
                edited_bytes
            ):

                logger.error(
                    "❌ OpenAI returned an invalid image."
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
                    "⚠️ Attempt %d failed watermark verification.",
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
            "🚫 Original media will NOT be posted."
        )

        return None

    except Exception:

        logger.exception(
            "❌ OPENAI WATERMARK PIPELINE FAILED"
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
# BACKWARD COMPATIBILITY
# ============================================================

async def regenerate_image_from_bytes(
    image_bytes: bytes,
):

    return await remove_watermarks_from_bytes(
        image_bytes
    )
