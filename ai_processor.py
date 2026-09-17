import os
import io
import re
import base64
import logging
import tempfile
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger("ai_processor")

# ============================================================
# CONFIG
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2").strip()
OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "low").strip().lower()

GENERATE_VARIATIONS = os.getenv("GENERATE_VARIATIONS", "true").strip().lower() in {
    "1", "true", "yes", "on"
}
VARIATION_COUNT = max(1, min(4, int(os.getenv("VARIATION_COUNT", "4"))))

NEW_MENTION = os.getenv("NEW_MENTION", "@PrimeAnalysiss").strip()
if NEW_MENTION and not NEW_MENTION.startswith("@"):
    NEW_MENTION = "@" + NEW_MENTION

# Caption replacements: do not rewrite anything else.
CAPTION_MENTIONS = (
    "@cappersfree",
    "@pickssman",
)

# Optional source-specific watermark style.
# Format:
# WATERMARK_PROFILES=-100111:cappersfree,-100222:pickssman,-100333:both
# If omitted, "both" is used so unknown/new channels are still supported.
PROFILE_RAW = os.getenv("WATERMARK_PROFILES", "").strip()
WATERMARK_PROFILES: Dict[int, str] = {}
if PROFILE_RAW:
    for item in PROFILE_RAW.replace(";", ",").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        channel_part, profile = item.split(":", 1)
        try:
            WATERMARK_PROFILES[int(channel_part.strip())] = profile.strip().lower()
        except ValueError:
            logger.warning("⚠️ Ignoring invalid WATERMARK_PROFILES entry: %r", item)

client = (
    AsyncOpenAI(api_key=OPENAI_API_KEY, max_retries=0)
    if OPENAI_API_KEY and AsyncOpenAI
    else None
)

# ============================================================
# CAPTIONS
# ============================================================

def replace_username(text: Optional[str]) -> Optional[str]:
    if not text:
        return text

    result = text.replace("*", "")
    for mention in CAPTION_MENTIONS:
        result = re.sub(
            re.escape(mention),
            NEW_MENTION,
            result,
            flags=re.IGNORECASE,
        )
    return result


async def rewrite_text(original_text: Optional[str]) -> Optional[str]:
    return replace_username(original_text)


# ============================================================
# SOURCE PROFILE
# ============================================================

def get_watermark_profile(source_channel: Optional[int]) -> str:
    if source_channel is not None:
        profile = WATERMARK_PROFILES.get(int(source_channel))
        if profile:
            return profile

    # Existing source channel is known to be Cappersfree.
    if source_channel == -1003593544389:
        return "cappersfree"

    # Unknown/new channels default to both watermark families.
    return "both"


# ============================================================
# LOCAL SIGNAL DETECTION
# ============================================================

def _to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def red_signal_ratio(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    strong1 = cv2.inRange(
        hsv,
        np.array([0, 120, 100], dtype=np.uint8),
        np.array([12, 255, 255], dtype=np.uint8),
    )
    strong2 = cv2.inRange(
        hsv,
        np.array([170, 120, 100], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )

    # Faint pink/red overlay.
    b, g, r = cv2.split(image)
    ri = r.astype(np.int16)
    gi = g.astype(np.int16)
    bi = b.astype(np.int16)
    faint = (
        (ri - gi >= 15)
        & (ri - bi >= 8)
        & (ri >= 175)
        & (gi <= 245)
    ).astype(np.uint8) * 255

    mask = strong1 | strong2 | faint
    return cv2.countNonZero(mask) / float(mask.shape[0] * mask.shape[1])


def blue_signal_ratio(image: np.ndarray) -> float:
    """Detect saturated blue/indigo watermark signal such as @pickssman."""
    if image is None or image.size == 0:
        return 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blue1 = cv2.inRange(
        hsv,
        np.array([95, 100, 60], dtype=np.uint8),
        np.array([135, 255, 255], dtype=np.uint8),
    )

    return cv2.countNonZero(blue1) / float(blue1.shape[0] * blue1.shape[1])


def green_signal_ratio(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array([35, 120, 60], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    return cv2.countNonZero(green) / float(green.shape[0] * green.shape[1])


def likely_has_watermark(
    image: Image.Image,
    source_channel: Optional[int] = None,
) -> bool:
    bgr = _to_bgr(image)
    red = red_signal_ratio(bgr)
    blue = blue_signal_ratio(bgr)
    green = green_signal_ratio(bgr)
    profile = get_watermark_profile(source_channel)

    logger.info(
        "🔎 Watermark signals | source=%s profile=%s red=%.4f%% blue=%.4f%% green=%.4f%%",
        source_channel,
        profile,
        red * 100,
        blue * 100,
        green * 100,
    )

    if profile == "cappersfree":
        return red >= 0.00015 or green >= 0.00008

    if profile == "pickssman":
        return blue >= 0.00008

    # both / auto
    return (
        red >= 0.00015
        or blue >= 0.00008
        or green >= 0.00008
    )


# ============================================================
# IMAGE API SIZING
# ============================================================

def _fit_for_image_api(
    source: Image.Image,
) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int, int, int]]:
    """Place the image on a supported standard canvas without cropping or stretching.

    GPT-Image-2 supports standard sizes such as 1024x1024, 1536x1024 and 1024x1536.
    Using these avoids sending tiny arbitrary canvases that can be rejected for
    insufficient pixel budget.
    """
    src = source.convert("RGB")
    original_size = src.size

    # First cap source dimensions while preserving aspect ratio.
    max_side = 1536
    if max(src.size) > max_side:
        scale = max_side / float(max(src.size))
        src = src.resize(
            (
                max(16, int(round(src.width * scale))),
                max(16, int(round(src.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )

    w, h = src.size
    ratio = w / float(h)

    if ratio > 1.25:
        canvas = (1536, 1024)
    elif ratio < 0.80:
        canvas = (1024, 1536)
    else:
        canvas = (1024, 1024)

    cw, ch = canvas
    scale = min(cw / float(w), ch / float(h))
    fw = max(16, int(round(w * scale)))
    fh = max(16, int(round(h * scale)))

    fitted = src.resize(
        (fw, fh),
        Image.Resampling.LANCZOS,
    )

    left = (cw - fw) // 2
    top = (ch - fh) // 2

    # Use black padding because it is easy to crop away after editing and
    # the prompt tells the model the padding is not part of the source.
    prepared = Image.new("RGB", canvas, (0, 0, 0))
    prepared.paste(fitted, (left, top))

    return prepared, original_size, (left, top, left + fw, top + fh)


def _restore_original_dimensions(
    edited: Image.Image,
    original_size: Tuple[int, int],
    crop_box: Tuple[int, int, int, int],
) -> Image.Image:
    x1, y1, x2, y2 = crop_box
    cropped = edited.crop(
        (
            x1,
            y1,
            min(x2, edited.width),
            min(y2, edited.height),
        )
    )

    if cropped.size != original_size:
        cropped = cropped.resize(
            original_size,
            Image.Resampling.LANCZOS,
        )

    return cropped


# ============================================================
# PROMPTS
# ============================================================

CAPPPERSFREE_INSTRUCTIONS = (
    "Remove every visible Cappersfree watermark and branding: "
    "@cappersfree text, faint/repeating cappersfree patterns, "
    "the CF circular/logo graphic, and any Cappersfree branding "
    "wherever it occurs in the image."
)

PICKSSMAN_INSTRUCTIONS = (
    "Remove every visible Pickssman watermark and branding: "
    "@pickssman text, blue/indigo Pickssman watermark text, "
    "and the t.me/cappersfree_247 watermark/link graphic wherever it occurs."
)

COMMON_PRESERVE_INSTRUCTIONS = (
    "Search the ENTIRE image; do not assume a fixed watermark location. "
    "If branding overlaps legitimate content, reconstruct the hidden pixels naturally. "
    "Preserve the exact source composition, framing, proportions, sports/betting interface, "
    "legitimate logos and icons, all legitimate text, numbers, scores, odds, names, faces, "
    "objects, colors, lighting and background. Do not crop, redesign, invent content, add "
    "text, add logos, or replace one watermark with another. Do not remove legitimate UI "
    "elements merely because they are red, blue or green. The only intended change is removal "
    "of the specified watermark/branding."
)


def build_clean_prompt(profile: str, motion: bool = False) -> str:
    if profile == "cappersfree":
        branding = CAPPPERSFREE_INSTRUCTIONS
    elif profile == "pickssman":
        branding = PICKSSMAN_INSTRUCTIONS
    else:
        branding = CAPPPERSFREE_INSTRUCTIONS + " " + PICKSSMAN_INSTRUCTIONS

    motion_note = ""
    if motion:
        motion_note = (
            " This is a representative frame extracted from an animation/video. "
            "Remove any pulsing, scaling, partially formed or fully formed CF logo from the frame."
        )

    return (
        "Create a CLEAN VERSION of this exact source image. "
        + branding
        + motion_note
        + " "
        + COMMON_PRESERVE_INSTRUCTIONS
    )


VARIATION_PROMPT = (
    "Create a subtle variation of this already-clean image. Preserve the same underlying "
    "information, layout, teams, odds, names, legitimate text, scores, logos, UI elements "
    "and important objects. Do not add, restore or invent @cappersfree, @pickssman, Cappersfree, "
    "Pickssman, CF, t.me/cappersfree_247, or any watermark/branding. Do not remove legitimate "
    "content. Keep the image immediately recognizable as the same source content."
)


# ============================================================
# OPENAI EDIT
# ============================================================

async def _openai_edit_one(
    image_bytes: bytes,
    prompt: str,
    filename: str,
) -> Optional[bytes]:
    if client is None:
        logger.error("❌ OPENAI_API_KEY missing or OpenAI package unavailable.")
        return None

    try:
        source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prepared, original_size, crop_box = _fit_for_image_api(source)

        image_buffer = io.BytesIO()
        prepared.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        image_buffer.name = "input.png"

        # Standard sizes are used for compatibility with the current GPT image API.
        requested_size = (
            "1536x1024"
            if prepared.size == (1536, 1024)
            else "1024x1536"
            if prepared.size == (1024, 1536)
            else "1024x1024"
        )

        logger.warning(
            "💰 OpenAI image edit | file=%s | quality=%s | size=%s",
            filename,
            OPENAI_IMAGE_QUALITY,
            requested_size,
        )

        response = await client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=image_buffer,
            prompt=prompt,
            quality=OPENAI_IMAGE_QUALITY,
            size=requested_size,
            output_format="png",
            n=1,
        )

        if not response.data:
            logger.error("❌ OpenAI returned no image data: %s", filename)
            return None

        b64 = getattr(response.data[0], "b64_json", None)
        if not b64:
            logger.error("❌ OpenAI response contained no b64_json: %s", filename)
            return None

        raw = base64.b64decode(b64)
        edited = Image.open(io.BytesIO(raw)).convert("RGB")
        restored = _restore_original_dimensions(
            edited,
            original_size,
            crop_box,
        )

        output = io.BytesIO()
        restored.save(output, format="PNG")
        cleaned = output.getvalue()

        logger.info(
            "✅ Clean image generated: %s | %d bytes",
            filename,
            len(cleaned),
        )
        return cleaned

    except Exception as exc:
        message = str(exc)
        if "insufficient_quota" in message or "credit_balance_exhausted" in message:
            logger.error("🛑 OpenAI account/project has no usable credit: %s", filename)
        elif getattr(exc, "status_code", None) == 429:
            logger.error("🛑 OpenAI rate limit: %s", filename)
        else:
            logger.exception("❌ OpenAI image edit failed: %s", filename)
        return None


# ============================================================
# STILL IMAGE CLEANUP
# ============================================================

async def remove_watermarks_from_bytes(
    image_bytes: bytes,
    filename: str = "image.jpg",
    source_channel: Optional[int] = None,
) -> Optional[bytes]:
    if not image_bytes:
        return None

    try:
        source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        logger.exception("❌ Could not open still image: %s", filename)
        return None

    if not likely_has_watermark(source, source_channel):
        logger.info(
            "✅ No convincing watermark signal found | source=%s | file=%s",
            source_channel,
            filename,
        )
        return None

    profile = get_watermark_profile(source_channel)
    prompt = build_clean_prompt(profile, motion=False)

    return await _openai_edit_one(
        image_bytes,
        prompt,
        filename,
    )


async def regenerate_image_from_bytes(
    image_bytes: bytes,
    filename: str = "image.jpg",
    source_channel: Optional[int] = None,
) -> Optional[bytes]:
    return await remove_watermarks_from_bytes(
        image_bytes,
        filename,
        source_channel,
    )


# ============================================================
# MOTION -> ONE CLEAN STILL
# ============================================================

def _frame_score(frame: Image.Image, source_channel: Optional[int]) -> float:
    bgr = _to_bgr(frame)
    red = red_signal_ratio(bgr)
    blue = blue_signal_ratio(bgr)
    green = green_signal_ratio(bgr)
    profile = get_watermark_profile(source_channel)

    if profile == "cappersfree":
        return red + 0.8 * green
    if profile == "pickssman":
        return blue
    return red + blue + 0.6 * green


def _pil_frame_to_png_bytes(frame: Image.Image) -> bytes:
    output = io.BytesIO()
    frame.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def _select_best_gif_frame(
    gif_bytes: bytes,
    source_channel: Optional[int],
) -> Optional[Image.Image]:
    try:
        source = Image.open(io.BytesIO(gif_bytes))
        count = getattr(source, "n_frames", 1)

        # Sample enough frames to catch a pulsing logo without requiring any API call.
        sample_count = min(30, count)
        indices = sorted(set(
            int(round(i * (count - 1) / max(1, sample_count - 1)))
            for i in range(sample_count)
        ))

        best_score = -1.0
        best = None

        for idx in indices:
            source.seek(idx)
            frame = source.convert("RGB").copy()
            score = _frame_score(frame, source_channel)
            if score > best_score:
                best_score = score
                best = frame

        logger.info(
            "🎞️ Selected GIF frame for cleanup | source=%s frames=%d score=%.6f",
            source_channel,
            count,
            best_score,
        )
        return best

    except Exception:
        logger.exception("❌ Could not inspect GIF frames")
        return None


def _select_best_video_frame(
    video_bytes: bytes,
    source_channel: Optional[int],
) -> Optional[Image.Image]:
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        ) as f:
            f.write(video_bytes)
            temp_path = f.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            logger.error("❌ Could not open video")
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return None

        sample_count = min(30, frame_count)
        indices = sorted(set(
            int(round(i * (frame_count - 1) / max(1, sample_count - 1)))
            for i in range(sample_count)
        ))

        best_score = -1.0
        best_frame = None

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            score = _frame_score(image, source_channel)

            if score > best_score:
                best_score = score
                best_frame = image

        cap.release()

        logger.info(
            "🎞️ Selected MP4 frame for cleanup | source=%s frames=%d score=%.6f",
            source_channel,
            frame_count,
            best_score,
        )

        return best_frame

    except Exception:
        logger.exception("❌ Could not extract representative MP4 frame")
        return None

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


async def remove_watermarks_from_gif_bytes(
    gif_bytes: bytes,
    filename: str = "animation.gif",
    source_channel: Optional[int] = None,
) -> Optional[bytes]:
    frame = _select_best_gif_frame(gif_bytes, source_channel)
    if frame is None:
        return None

    if not likely_has_watermark(frame, source_channel):
        logger.info("✅ GIF frame has no convincing watermark signal")
        return None

    return await _openai_edit_one(
        _pil_frame_to_png_bytes(frame),
        build_clean_prompt(
            get_watermark_profile(source_channel),
            motion=True,
        ),
        filename,
    )


async def remove_watermarks_from_video_bytes(
    video_bytes: bytes,
    filename: str = "video.mp4",
    source_channel: Optional[int] = None,
) -> Optional[bytes]:
    frame = _select_best_video_frame(video_bytes, source_channel)
    if frame is None:
        return None

    if not likely_has_watermark(frame, source_channel):
        logger.info("✅ MP4 frame has no convincing watermark signal")
        return None

    return await _openai_edit_one(
        _pil_frame_to_png_bytes(frame),
        build_clean_prompt(
            get_watermark_profile(source_channel),
            motion=True,
        ),
        filename,
    )


# ============================================================
# FOUR VARIATIONS
# ============================================================

async def generate_variations_from_clean_image(
    clean_image_bytes: bytes,
    filename: str = "clean.png",
    count: Optional[int] = None,
) -> List[bytes]:
    """Generate up to four subtle variations from an already-clean image."""
    if not GENERATE_VARIATIONS:
        return [clean_image_bytes]

    if client is None:
        return [clean_image_bytes]

    count = max(1, min(4, count or VARIATION_COUNT))

    try:
        source = Image.open(io.BytesIO(clean_image_bytes)).convert("RGB")
        prepared, original_size, crop_box = _fit_for_image_api(source)

        image_buffer = io.BytesIO()
        prepared.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        image_buffer.name = "clean.png"

        requested_size = (
            "1536x1024"
            if prepared.size == (1536, 1024)
            else "1024x1536"
            if prepared.size == (1024, 1536)
            else "1024x1024"
        )

        logger.info(
            "💰 OpenAI variations | file=%s | count=%d | quality=%s | size=%s",
            filename,
            count,
            OPENAI_IMAGE_QUALITY,
            requested_size,
        )

        response = await client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=image_buffer,
            prompt=VARIATION_PROMPT,
            quality=OPENAI_IMAGE_QUALITY,
            size=requested_size,
            output_format="png",
            n=count,
        )

        results: List[bytes] = []

        for item in response.data or []:
            b64 = getattr(item, "b64_json", None)
            if not b64:
                continue

            try:
                raw = base64.b64decode(b64)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                restored = _restore_original_dimensions(
                    img,
                    original_size,
                    crop_box,
                )

                out = io.BytesIO()
                restored.save(out, format="PNG")
                results.append(out.getvalue())

            except Exception:
                logger.exception("❌ Failed decoding one variation")

        logger.info(
            "✅ Generated %d variations for %s",
            len(results),
            filename,
        )

        return results or [clean_image_bytes]

    except Exception:
        logger.exception(
            "❌ Variation generation failed: %s",
            filename,
        )
        return [clean_image_bytes]
