import os
import io
import re
import base64
import asyncio
import logging
import tempfile
import subprocess

from pathlib import Path

from PIL import Image
from openai import OpenAI, RateLimitError
from aiogram.types import BufferedInputFile


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("ai_processor")


# ============================================================
# OPENAI ENVIRONMENT
# ============================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
).strip()

if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY is missing.")


# IMPORTANT:
# max_retries=0 prevents the SDK from repeatedly retrying
# requests that cannot succeed, such as exhausted credits or
# invalid parameters.
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
# OPENAI IMAGE CONFIGURATION
# ============================================================

OPENAI_IMAGE_MODEL = os.getenv(
    "OPENAI_IMAGE_MODEL",
    "gpt-image-2",
).strip()

OPENAI_IMAGE_QUALITY = os.getenv(
    "OPENAI_IMAGE_QUALITY",
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
# VERIFICATION CONFIGURATION
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
    "🎞️ GIF/MP4 workflow: representative still"
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
# FRAME SCORING
# ============================================================

def score_frame(
    image: Image.Image,
) -> float:
    """
    This is ONLY used to choose a representative frame.

    It is NOT a watermark detector.

    Lower score is preferred.
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
# MP4 / TELEGRAM GIF FRAME EXTRACTION
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
            # Use a simple valid FFmpeg filter.
            #
            # Do NOT use:
            #     fps=min(12,1000)
            #
            # because FFmpeg interprets that incorrectly.
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

            logger.info(
                "🎞️ Running FFmpeg frame extraction..."
            )

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
                    )[-5000:]
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
                        "⚠️ Could not read frame %s",
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
# PREPARE STILL
# ============================================================

def prepare_still_image(
    image_bytes: bytes,
):

    if not image_bytes:

        return None

    # --------------------------------------------------------
    # GIF
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MP4 / video
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Normal image
    # --------------------------------------------------------

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

1. every visible, faint, translucent, partial, faded, or
   stylized "@cappersfree" / "cappersfree" watermark

2. every CF graphic/logo associated with that watermark

This is NOT a redesign and NOT a full recreation.

Reconstruct the background/content underneath the removed
watermark naturally so the result looks clean and professional.

PRESERVE AS MUCH OF THE SOURCE AS POSSIBLE:

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
- unrelated icons
- typography
- font appearance
- colors
- gradients
- shadows
- panels
- borders
- spacing
- composition
- layout
- aspect ratio

DO NOT:

- crop
- resize unnecessarily
- redesign
- rewrite legitimate text
- change legitimate numbers
- change scores
- change odds
- change names
- change players
- change faces
- add another logo
- add another username
- add branding
- add a watermark
- replace the CF watermark with another graphic
- blur the removed area
- leave a visible patch or blank region

The ONLY requested modification is:

REMOVE THE CF GRAPHIC/LOGO AND @CAPPERSFREE WATERMARK.

Everything else should remain visually faithful to the source.
"""


WATERMARK_RETRY_PROMPT = """
The previous edited result was rejected because the watermark
was still visible or a faint/ghosted version may remain.

Perform another careful cleanup of the ENTIRE image.

Look for all versions of:

- @cappersfree
- cappersfree
- faded @cappersfree
- translucent @cappersfree
- partial cappersfree
- CF logo
- faded CF logo
- translucent CF logo
- ghosted remnants

Remove all of them.

Reconstruct the underlying background naturally.

Do NOT alter legitimate:
- text
- numbers
- scores
- odds
- names
- players
- faces
- colors
- typography
- panels
- layout
- composition

Do not redesign the image.
"""


WATERMARK_VERIFICATION_PROMPT = """
You are the final quality-control checker for a watermark
removal pipeline.

Inspect the provided edited image.

The unwanted watermark is specifically:

- CF graphic/logo
- @cappersfree
- cappersfree
- faint versions
- translucent versions
- partial versions
- ghosted remnants

DO NOT classify legitimate sports graphics as a watermark.

Reply with EXACTLY:

CLEAN

or

NOT_CLEAN

Reply NOT_CLEAN if:
- any CF logo remains
- any cappersfree text remains
- any faint watermark remains
- any ghosted watermark remains
- you are uncertain

Reply CLEAN only when the target watermark is genuinely absent.
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
        #
        # Do NOT pass:
        #     input_fidelity
        #     moderation
        #
        # They caused API 400 errors in the deployed
        # version.
        response = (
            openai_client
            .images
            .edit(
                model=OPENAI_IMAGE_MODEL,
                image=image_file,
                prompt=prompt,
                size="auto",
                quality=OPENAI_IMAGE_QUALITY,
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
        # Exhausted API credits.
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
                "❌ This is a billing/quota issue, "
                "not an image-processing issue."
            )

            logger.error(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            return (
                "__OPENAI_CREDITS_EXHAUSTED__"
            )

        # Other rate limits.
        logger.error(
            "❌ OpenAI rate limit error: %s",
            exc,
        )

        return None

    except Exception as exc:

        status_code = getattr(
            exc,
            "status_code",
            None,
        )

        error_text = str(
            exc
        )

        logger.error(
            "❌ OpenAI image edit failed."
        )

        logger.error(
            "❌ Exception type: %s",
            type(exc).__name__,
        )

        logger.error(
            "❌ Status: %s",
            status_code,
        )

        logger.error(
            "❌ Error: %s",
            error_text,
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

        # ----------------------------------------------------
        # 400 = invalid request/configuration.
        #
        # Do not retry an identical invalid request.
        # ----------------------------------------------------

        if status_code == 400:

            logger.error(
                "❌ OpenAI rejected the request parameters."
            )

            return (
                "__OPENAI_BAD_REQUEST__"
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
            "❌ Failed to decode OpenAI image."
        )

        return None

    logger.info(
        "✅ OpenAI returned %d bytes.",
        len(output_bytes),
    )

    return output_bytes


# ============================================================
# VERIFICATION
# ============================================================

def verify_watermark_is_removed(
    image_bytes: bytes,
):

    if not WATERMARK_VERIFICATION_ENABLED:

        logger.warning(
            "⚠️ Watermark verification disabled."
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
        # STEP 1 — PREPARE STILL
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
            # Account has no credits.
            # ------------------------------------------------

            if (
                edited_bytes
                == "__OPENAI_CREDITS_EXHAUSTED__"
            ):

                logger.error(
                    "🚫 Stopping because OpenAI API "
                    "credits are exhausted."
                )

                return None

            # ------------------------------------------------
            # Invalid request parameters.
            # ------------------------------------------------

            if (
                edited_bytes
                == "__OPENAI_BAD_REQUEST__"
            ):

                logger.error(
                    "🚫 Stopping because the OpenAI "
                    "edit request was rejected."
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

            logger.info(
                "3️⃣ Verifying edited image..."
            )

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

        # ====================================================
        # NOTHING VERIFIED
        # ====================================================

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
