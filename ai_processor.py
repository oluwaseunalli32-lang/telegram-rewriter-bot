import os
import io
import re
import base64
import logging
from typing import Optional, List, Tuple

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
# Low is the safest beta default for cost. gpt-image-2 supports low/medium/high.
OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "low").strip().lower()

# Hard safety limit. Counted per returned image, not just per HTTP request.
BETA_MODE = os.getenv("BETA_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
MAX_OPENAI_IMAGES_PER_RUN = int(os.getenv("MAX_OPENAI_IMAGES_PER_RUN", "1"))

# Optional four-variation stage. Keep OFF during beta.
GENERATE_VARIATIONS = os.getenv("GENERATE_VARIATIONS", "false").strip().lower() in {"1", "true", "yes", "on"}
VARIATION_COUNT = max(1, min(4, int(os.getenv("VARIATION_COUNT", "4"))))
MAX_VARIATION_IMAGES_PER_RUN = int(os.getenv("MAX_VARIATION_IMAGES_PER_RUN", "4"))

NEW_MENTION = os.getenv("NEW_MENTION", "@PrimeAnalysiss").strip()
if NEW_MENTION and not NEW_MENTION.startswith("@"):
    NEW_MENTION = "@" + NEW_MENTION

OLD_MENTION = "@cappersfree"

_openai_images_used = 0

client = (
    AsyncOpenAI(api_key=OPENAI_API_KEY)
    if OPENAI_API_KEY and AsyncOpenAI
    else None
)


# ============================================================
# CAPTIONS
# ============================================================

def replace_username(text: Optional[str]) -> Optional[str]:
    """Caption rule: remove * and replace only @cappersfree."""
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


async def rewrite_text(original_text: Optional[str]) -> Optional[str]:
    return replace_username(original_text)


# ============================================================
# LOCAL WATERMARK SIGNAL DETECTION
# ============================================================

def _bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def red_signal_ratio(image: np.ndarray) -> float:
    """Detect strong or faint red/pink signal anywhere in an image.

    This is intentionally only a trigger, not the edit mask. The OpenAI
    image editor is asked to locate the branding across the entire image.
    """
    if image is None or image.size == 0:
        return 0.0

    hsv = _bgr_to_hsv(image)

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

    # Faint pink/red overlay: modest red dominance with reasonably bright pixels.
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
    ratio = cv2.countNonZero(mask) / float(mask.shape[0] * mask.shape[1])
    return ratio


def green_signal_ratio(image: np.ndarray) -> float:
    """Detect green signal used by the animated CF logo."""
    if image is None or image.size == 0:
        return 0.0

    hsv = _bgr_to_hsv(image)
    green = cv2.inRange(
        hsv,
        np.array([35, 120, 60], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )
    return cv2.countNonZero(green) / float(green.shape[0] * green.shape[1])


def likely_has_cappersfree_watermark(image: Image.Image) -> bool:
    """Conservative gate for whether an OpenAI edit is worth attempting."""
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    red_ratio = red_signal_ratio(bgr)
    green_ratio = green_signal_ratio(bgr)

    logger.info(
        "🔎 Watermark trigger signals | red=%.4f%% green=%.4f%%",
        red_ratio * 100,
        green_ratio * 100,
    )

    # The source branding is known to include strong red text and/or a bright green CF logo.
    return red_ratio >= 0.00015 or green_ratio >= 0.00008


# ============================================================
# IMAGE SIZING / PADDING
# ============================================================

def _fit_for_image_api(source: Image.Image) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int, int, int]]:
    """Prepare an image for GPT image editing while preserving the full original frame.

    GPT image endpoints require a bounded aspect ratio. If an incoming image is
    outside 3:1, edge-padding is used instead of cropping. The padded result is
    cropped back to the original content after editing.
    """
    src = source.convert("RGB")
    original_size = src.size

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

    # Pad only when outside the allowed 1:3..3:1 range.
    if ratio > 3.0:
        new_h = int(np.ceil(w / 3.0 / 16.0) * 16)
        pad_total = max(0, new_h - h)
        top = pad_total // 2
        bottom = pad_total - top
        padded = Image.new("RGB", (w, new_h))
        padded.paste(src, (0, top))
        crop_box = (0, top, w, top + h)
        src = padded
    elif ratio < (1.0 / 3.0):
        new_w = int(np.ceil(h / 3.0 / 16.0) * 16)
        pad_total = max(0, new_w - w)
        left = pad_total // 2
        right = pad_total - left
        padded = Image.new("RGB", (new_w, h))
        padded.paste(src, (left, 0))
        crop_box = (left, 0, left + w, h)
        src = padded
    else:
        crop_box = (0, 0, src.width, src.height)

    # API wants dimensions divisible by 16. Pad minimally instead of cropping.
    extra_w = (16 - (src.width % 16)) % 16
    extra_h = (16 - (src.height % 16)) % 16

    if extra_w or extra_h:
        padded = Image.new("RGB", (src.width + extra_w, src.height + extra_h))
        padded.paste(src, (0, 0))
        src = padded
        # Any bottom/right padding should not be part of the desired content.
        crop_box = (crop_box[0], crop_box[1], crop_box[2], crop_box[3])

    return src, original_size, crop_box


def _restore_original_dimensions(
    edited: Image.Image,
    original_size: Tuple[int, int],
    crop_box: Tuple[int, int, int, int],
) -> Image.Image:
    x1, y1, x2, y2 = crop_box
    cropped = edited.crop((x1, y1, min(x2, edited.width), min(y2, edited.height)))
    if cropped.size != original_size:
        cropped = cropped.resize(original_size, Image.Resampling.LANCZOS)
    return cropped


# ============================================================
# OPENAI EDIT PROMPTS
# ============================================================

CLEAN_PROMPT = (
    "Edit this exact source image into a CLEAN VERSION. Search the ENTIRE image, "
    "not a fixed location, for Cappersfree branding and remove ONLY that branding. "
    "Remove every visible @cappersfree watermark, including solid bright red text, "
    "faint/repeating translucent cappersfree text or patterns, and any Cappersfree/CF "
    "circular logo or graphic. If the watermark overlaps legitimate content, reconstruct "
    "the hidden pixels so the original content looks natural. Preserve the original "
    "composition, framing, proportions, sports/betting UI, legitimate logos and icons, "
    "all non-watermark text, numbers, scores, odds, names, faces, objects, colors, "
    "lighting and background. Do not crop. Do not redesign. Do not add any text or logo. "
    "Do not replace the watermark with another brand. Do not alter legitimate red text or "
    "graphics merely because they are red. The only intended change is complete removal "
    "of Cappersfree watermark/branding and natural reconstruction underneath it."
)

VARIATION_PROMPT = (
    "Create a tasteful variation of this already-clean source image while preserving its "
    "core information and composition. Keep all legitimate text, numbers, teams, odds, "
    "UI elements and important objects readable and intact. Do not add any Cappersfree, "
    "@cappersfree, CF logo, watermark, branding, or replacement text. Make only subtle "
    "visual variations in presentation while keeping the same underlying content."
)


# ============================================================
# OPENAI REQUEST
# ============================================================

async def _openai_edit_one(
    image_bytes: bytes,
    prompt: str,
    filename: str,
) -> Optional[bytes]:
    global _openai_images_used

    if client is None:
        logger.warning("⚠️ OPENAI_API_KEY missing. Skipping AI edit.")
        return None

    if _openai_images_used >= MAX_OPENAI_IMAGES_PER_RUN:
        logger.warning(
            "🛑 OpenAI image limit reached: %d/%d. Keeping original.",
            _openai_images_used,
            MAX_OPENAI_IMAGES_PER_RUN,
        )
        return None

    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prepared, original_size, crop_box = _fit_for_image_api(source)

    image_buffer = io.BytesIO()
    prepared.save(image_buffer, format="PNG")
    image_buffer.seek(0)
    image_buffer.name = "input.png"

    requested_size = f"{prepared.width}x{prepared.height}"

    logger.warning(
        "💰 OpenAI image edit %d/%d | file=%s | quality=%s | size=%s",
        _openai_images_used + 1,
        MAX_OPENAI_IMAGES_PER_RUN,
        filename,
        OPENAI_IMAGE_QUALITY,
        requested_size,
    )

    try:
        response = await client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=image_buffer,
            prompt=prompt,
            quality=OPENAI_IMAGE_QUALITY,
            size=requested_size,
            output_format="png",
            n=1,
        )
    except Exception:
        logger.exception("❌ OpenAI image edit failed: %s", filename)
        return None

    # This limit is on actual returned images, so a successful response is counted here.
    _openai_images_used += 1

    if not response.data:
        logger.error("❌ OpenAI returned no image data: %s", filename)
        return None

    b64 = getattr(response.data[0], "b64_json", None)
    if not b64:
        logger.error("❌ OpenAI response contained no b64_json: %s", filename)
        return None

    try:
        edited_bytes = base64.b64decode(b64)
        edited_img = Image.open(io.BytesIO(edited_bytes)).convert("RGB")
        restored = _restore_original_dimensions(
            edited_img,
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

    except Exception:
        logger.exception("❌ Could not decode/restore OpenAI image: %s", filename)
        return None


# ============================================================
# STILL IMAGE CLEANUP
# ============================================================

async def remove_watermarks_from_bytes(
    image_bytes: bytes,
    filename: str = "image.jpg",
) -> Optional[bytes]:
    """Clean a still image using one OpenAI edit at most during beta."""
    if not image_bytes:
        return None

    try:
        source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        logger.exception("❌ Could not open still image: %s", filename)
        return None

    if not likely_has_cappersfree_watermark(source):
        logger.info(
            "✅ Local detector found no convincing Cappersfree signal: %s",
            filename,
        )
        return None

    return await _openai_edit_one(
        image_bytes,
        CLEAN_PROMPT,
        filename,
    )


async def regenerate_image_from_bytes(
    image_bytes: bytes,
    filename: str = "image.jpg",
) -> Optional[bytes]:
    return await remove_watermarks_from_bytes(image_bytes, filename)


# ============================================================
# MOTION -> CLEAN STILL IMAGE
# ============================================================

def _pil_frame_to_png_bytes(frame: Image.Image) -> bytes:
    output = io.BytesIO()
    frame.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def _select_best_gif_frame(gif_bytes: bytes) -> Optional[Image.Image]:
    """Choose a frame where the watermark is most visible, so one edit can remove it."""
    try:
        source = Image.open(io.BytesIO(gif_bytes))
        count = getattr(source, "n_frames", 1)
        best_score = -1.0
        best = None

        for idx in range(count):
            source.seek(idx)
            frame = source.convert("RGB")
            arr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

            red = red_signal_ratio(arr)
            green = green_signal_ratio(arr)

            # Favor frames with the clearest combined watermark signal.
            score = red * 1.0 + green * 0.8
            # Middle-ish frames are also preferable to transition frames.
            mid_penalty = abs((idx / max(1, count - 1)) - 0.55)
            score -= mid_penalty * 0.0002

            if score > best_score:
                best_score = score
                best = frame.copy()

        if best is None:
            source.seek(max(0, count // 2))
            best = source.convert("RGB")

        logger.info(
            "🎞️ Selected GIF frame for cleanup | frames=%d | score=%.6f",
            count,
            best_score,
        )
        return best

    except Exception:
        logger.exception("❌ Could not inspect GIF frames")
        return None


def _select_best_video_frame(video_bytes: bytes) -> Optional[Image.Image]:
    """Choose a representative frame with the strongest watermark signal."""
    import tempfile

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(video_bytes)
            temp_path = f.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            logger.error("❌ Could not open video for representative-frame extraction")
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return None

        sample_indices = sorted(set([
            int(frame_count * 0.30),
            int(frame_count * 0.45),
            int(frame_count * 0.55),
            int(frame_count * 0.65),
            int(frame_count * 0.75),
        ]))

        best_score = -1.0
        best_frame = None

        for idx in sample_indices:
            idx = max(0, min(frame_count - 1, idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue

            red = red_signal_ratio(frame)
            green = green_signal_ratio(frame)
            score = red * 1.0 + green * 0.8

            if score > best_score:
                best_score = score
                best_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cap.release()

        if best_frame is None:
            return None

        logger.info(
            "🎞️ Selected MP4 frame for cleanup | frames=%d | score=%.6f",
            frame_count,
            best_score,
        )
        return Image.fromarray(best_frame)

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
) -> Optional[bytes]:
    """Convert GIF to ONE clean still image. No GIF is returned."""
    frame = _select_best_gif_frame(gif_bytes)
    if frame is None:
        return None

    frame_bytes = _pil_frame_to_png_bytes(frame)

    # For the beta, treat the GIF as an intentional watermark-bearing item.
    # Still use the local gate before spending an OpenAI image edit.
    try:
        if not likely_has_cappersfree_watermark(frame):
            logger.info("✅ GIF representative frame has no convincing watermark signal")
            return None
    except Exception:
        return None

    return await _openai_edit_one(
        frame_bytes,
        CLEAN_PROMPT + " This source came from an animation; remove the CF logo even if it is a pulsing/scaling frame artifact.",
        filename,
    )


async def remove_watermarks_from_video_bytes(
    video_bytes: bytes,
    filename: str = "video.mp4",
) -> Optional[bytes]:
    """Convert MP4 to ONE clean still PNG. No animation is returned."""
    frame = _select_best_video_frame(video_bytes)
    if frame is None:
        return None

    frame_bytes = _pil_frame_to_png_bytes(frame)

    if not likely_has_cappersfree_watermark(frame):
        logger.info("✅ MP4 representative frame has no convincing watermark signal")
        return None

    return await _openai_edit_one(
        frame_bytes,
        CLEAN_PROMPT + " This source came from a video/animation; remove any visible or partially-formed pulsing CF logo completely.",
        filename,
    )


# ============================================================
# OPTIONAL FOUR-VARIATION STAGE
# ============================================================

async def generate_variations_from_clean_image(
    clean_image_bytes: bytes,
    filename: str = "clean.png",
    count: Optional[int] = None,
) -> List[bytes]:
    """Optional second stage. Uses one edit request with n=count.

    Each returned image consumes one image output from the account, so this is
    intentionally disabled unless GENERATE_VARIATIONS=true.
    """
    global _openai_images_used

    if not GENERATE_VARIATIONS:
        return [clean_image_bytes]

    if client is None:
        return [clean_image_bytes]

    count = max(1, min(4, count or VARIATION_COUNT))
    available = MAX_VARIATION_IMAGES_PER_RUN
    count = min(count, available)

    if BETA_MODE:
        remaining = max(0, MAX_OPENAI_IMAGES_PER_RUN - _openai_images_used)
        count = min(count, remaining)

    if count <= 0:
        logger.warning("🛑 No OpenAI image budget remains for variations.")
        return [clean_image_bytes]

    try:
        source = Image.open(io.BytesIO(clean_image_bytes)).convert("RGB")
        prepared, original_size, crop_box = _fit_for_image_api(source)

        image_buffer = io.BytesIO()
        prepared.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        image_buffer.name = "clean.png"

        response = await client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=image_buffer,
            prompt=VARIATION_PROMPT,
            quality=OPENAI_IMAGE_QUALITY,
            size=f"{prepared.width}x{prepared.height}",
            output_format="png",
            n=count,
        )

        returned: List[bytes] = []
        for item in response.data or []:
            b64 = getattr(item, "b64_json", None)
            if not b64:
                continue

            try:
                raw = base64.b64decode(b64)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                restored = _restore_original_dimensions(img, original_size, crop_box)
                out = io.BytesIO()
                restored.save(out, format="PNG")
                returned.append(out.getvalue())
            except Exception:
                logger.exception("❌ Failed decoding one variation")

        _openai_images_used += len(returned)

        if returned:
            logger.info(
                "✅ Generated %d optional variations for %s",
                len(returned),
                filename,
            )
            return returned

    except Exception:
        logger.exception("❌ Variation generation failed: %s", filename)

    return [clean_image_bytes]
