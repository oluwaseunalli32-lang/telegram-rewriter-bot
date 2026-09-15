import os
import io
import re
import base64
import logging
import tempfile
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
OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "low").strip().lower()

# Production mode: NO application-imposed image/hour/run limit.
# OpenAI billing, account limits, and API rate limits still apply.
GENERATE_VARIATIONS = os.getenv("GENERATE_VARIATIONS", "true").strip().lower() in {"1", "true", "yes", "on"}
VARIATION_COUNT = max(1, min(4, int(os.getenv("VARIATION_COUNT", "4"))))

NEW_MENTION = os.getenv("NEW_MENTION", "@PrimeAnalysiss").strip()
if NEW_MENTION and not NEW_MENTION.startswith("@"):
    NEW_MENTION = "@" + NEW_MENTION
OLD_MENTION = "@cappersfree"

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
    return re.sub(re.escape(OLD_MENTION), NEW_MENTION, result, flags=re.IGNORECASE)


async def rewrite_text(original_text: Optional[str]) -> Optional[str]:
    return replace_username(original_text)


# ============================================================
# WATERMARK SIGNALS
# ============================================================

def red_signal_ratio(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    strong1 = cv2.inRange(hsv, np.array([0, 120, 100], np.uint8), np.array([12, 255, 255], np.uint8))
    strong2 = cv2.inRange(hsv, np.array([170, 120, 100], np.uint8), np.array([179, 255, 255], np.uint8))
    b, g, r = cv2.split(image)
    ri, gi, bi = r.astype(np.int16), g.astype(np.int16), b.astype(np.int16)
    faint = ((ri - gi >= 15) & (ri - bi >= 8) & (ri >= 175) & (gi <= 245)).astype(np.uint8) * 255
    mask = strong1 | strong2 | faint
    return cv2.countNonZero(mask) / float(mask.shape[0] * mask.shape[1])


def green_signal_ratio(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 120, 60], np.uint8), np.array([95, 255, 255], np.uint8))
    return cv2.countNonZero(green) / float(green.shape[0] * green.shape[1])


def likely_has_cappersfree_watermark(image: Image.Image) -> bool:
    arr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    red = red_signal_ratio(arr)
    green = green_signal_ratio(arr)
    logger.info("🔎 Watermark trigger signals | red=%.4f%% green=%.4f%%", red * 100, green * 100)
    return red >= 0.00015 or green >= 0.00008


# ============================================================
# API IMAGE SIZING
# ============================================================

def _fit_for_image_api(source: Image.Image) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int, int, int]]:
    src = source.convert("RGB")
    original_size = src.size
    max_source_side = 1536

    if max(src.size) > max_source_side:
        scale = max_source_side / float(max(src.size))
        src = src.resize((max(16, int(round(src.width * scale))), max(16, int(round(src.height * scale)))), Image.Resampling.LANCZOS)

    w, h = src.size
    ratio = w / float(h)
    if ratio > 1.15:
        canvas = (1536, 1024)
    elif ratio < 0.87:
        canvas = (1024, 1536)
    else:
        canvas = (1024, 1024)

    cw, ch = canvas
    scale = min(cw / float(w), ch / float(h))
    fw, fh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
    fitted = src.resize((fw, fh), Image.Resampling.LANCZOS) if (fw, fh) != (w, h) else src
    left, top = (cw - fw) // 2, (ch - fh) // 2

    prepared = Image.new("RGB", canvas, (0, 0, 0))
    prepared.paste(fitted, (left, top))
    return prepared, original_size, (left, top, left + fw, top + fh)


def _restore_original_dimensions(edited: Image.Image, original_size: Tuple[int, int], crop_box: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = crop_box
    cropped = edited.crop((x1, y1, min(x2, edited.width), min(y2, edited.height)))
    if cropped.size != original_size:
        cropped = cropped.resize(original_size, Image.Resampling.LANCZOS)
    return cropped


# ============================================================
# PROMPTS
# ============================================================

CLEAN_PROMPT = (
    "Create a clean version of this exact source image. Search the ENTIRE image, "
    "regardless of watermark position, for all Cappersfree branding and remove only that branding. "
    "Remove every visible @cappersfree watermark, including solid bright-red text, "
    "faint/repeating translucent cappersfree text/patterns, and every Cappersfree/CF logo or graphic. "
    "If branding overlaps legitimate content, reconstruct the hidden pixels naturally from surrounding context. "
    "Preserve the exact original composition, framing, proportions, sports/betting interface, "
    "legitimate logos and icons, all legitimate text, numbers, scores, odds, names, faces, objects, "
    "colors, lighting, and background. Do not crop, redesign, add text, add logos, or replace the watermark "
    "with another brand. Do not modify legitimate red or green interface elements merely because of their color. "
    "The only intended change is complete removal of Cappersfree branding and natural reconstruction underneath it."
)

VARIATION_PROMPT = (
    "Create a subtle visual variation of this ALREADY CLEAN image. Keep the same underlying information, "
    "layout, teams, odds, names, legitimate text, scores, UI elements and important objects. "
    "Do not add, restore or invent @cappersfree, Cappersfree, CF, or any watermark/branding. "
    "Do not remove legitimate content. Keep the image immediately recognizable as the same source content."
)


# ============================================================
# OPENAI EDIT
# ============================================================

async def _openai_edit_one(image_bytes: bytes, prompt: str, filename: str) -> Optional[bytes]:
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

        logger.warning(
            "💰 OpenAI image edit | file=%s | quality=%s | size=%sx%s",
            filename,
            OPENAI_IMAGE_QUALITY,
            prepared.width,
            prepared.height,
        )

        response = await client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=image_buffer,
            prompt=prompt,
            quality=OPENAI_IMAGE_QUALITY,
            size=f"{prepared.width}x{prepared.height}",
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
        restored = _restore_original_dimensions(edited, original_size, crop_box)

        output = io.BytesIO()
        restored.save(output, format="PNG")
        cleaned = output.getvalue()

        logger.info("✅ Clean image generated: %s | %d bytes", filename, len(cleaned))
        return cleaned

    except Exception as exc:
        status = getattr(exc, "status_code", None)
        message = str(exc)
        if status == 429 or "insufficient_quota" in message or "credit_balance_exhausted" in message:
            logger.error("🛑 OpenAI rejected the edit because of quota/rate limits: %s", filename)
        else:
            logger.exception("❌ OpenAI image edit failed: %s", filename)
        return None


# ============================================================
# STILL IMAGE
# ============================================================

async def remove_watermarks_from_bytes(image_bytes: bytes, filename: str = "image.jpg") -> Optional[bytes]:
    if not image_bytes:
        return None

    try:
        source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        logger.exception("❌ Could not open still image: %s", filename)
        return None

    # Gate only skips obviously clean images. It is NOT location-based.
    if not likely_has_cappersfree_watermark(source):
        logger.info("✅ No convincing Cappersfree signal found: %s", filename)
        return None

    return await _openai_edit_one(image_bytes, CLEAN_PROMPT, filename)


async def regenerate_image_from_bytes(image_bytes: bytes, filename: str = "image.jpg") -> Optional[bytes]:
    return await remove_watermarks_from_bytes(image_bytes, filename)


# ============================================================
# MOTION -> ONE CLEAN STILL
# ============================================================

def _pil_frame_to_png_bytes(frame: Image.Image) -> bytes:
    buf = io.BytesIO()
    frame.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _score_frame(frame: Image.Image) -> float:
    arr = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
    return red_signal_ratio(arr) + 0.8 * green_signal_ratio(arr)


def _select_best_gif_frame(gif_bytes: bytes) -> Optional[Image.Image]:
    try:
        source = Image.open(io.BytesIO(gif_bytes))
        count = getattr(source, "n_frames", 1)
        candidates = sorted(set([0, count // 4, count // 2, max(0, int(count * 0.75)), count - 1]))
        best_score, best = -1.0, None
        for idx in candidates:
            source.seek(idx)
            frame = source.convert("RGB").copy()
            score = _score_frame(frame)
            if score > best_score:
                best_score, best = score, frame
        logger.info("🎞️ Selected GIF frame for cleanup | frames=%d | score=%.6f", count, best_score)
        return best
    except Exception:
        logger.exception("❌ Could not inspect GIF frames")
        return None


def _select_best_video_frame(video_bytes: bytes) -> Optional[Image.Image]:
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(video_bytes)
            path = f.name
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.error("❌ Could not open video")
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        candidates = sorted(set([int(total * p) for p in (0.15, 0.30, 0.45, 0.55, 0.65, 0.75, 0.85)]))
        best_score, best = -1.0, None
        for idx in candidates:
            idx = max(0, min(total - 1, idx))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            score = _score_frame(image)
            if score > best_score:
                best_score, best = score, image
        cap.release()
        logger.info("🎞️ Selected MP4 frame for cleanup | frames=%d | score=%.6f", total, best_score)
        return best
    except Exception:
        logger.exception("❌ Could not extract representative MP4 frame")
        return None
    finally:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass


async def remove_watermarks_from_gif_bytes(gif_bytes: bytes, filename: str = "animation.gif") -> Optional[bytes]:
    frame = _select_best_gif_frame(gif_bytes)
    if frame is None or not likely_has_cappersfree_watermark(frame):
        return None
    return await _openai_edit_one(
        _pil_frame_to_png_bytes(frame),
        CLEAN_PROMPT + " This is a representative frame from an animation. Remove the pulsing/scaling CF logo completely.",
        filename,
    )


async def remove_watermarks_from_video_bytes(video_bytes: bytes, filename: str = "video.mp4") -> Optional[bytes]:
    frame = _select_best_video_frame(video_bytes)
    if frame is None or not likely_has_cappersfree_watermark(frame):
        return None
    return await _openai_edit_one(
        _pil_frame_to_png_bytes(frame),
        CLEAN_PROMPT + " This is a representative frame from a video/animation. Remove the pulsing/scaling CF logo completely.",
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
    if not GENERATE_VARIATIONS or client is None:
        return [clean_image_bytes]

    count = max(1, min(4, count or VARIATION_COUNT))

    try:
        source = Image.open(io.BytesIO(clean_image_bytes)).convert("RGB")
        prepared, original_size, crop_box = _fit_for_image_api(source)
        image_buffer = io.BytesIO()
        prepared.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        image_buffer.name = "clean.png"

        logger.warning(
            "💰 OpenAI variations | file=%s | count=%d | quality=%s",
            filename,
            count,
            OPENAI_IMAGE_QUALITY,
        )

        response = await client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=image_buffer,
            prompt=VARIATION_PROMPT,
            quality=OPENAI_IMAGE_QUALITY,
            size=f"{prepared.width}x{prepared.height}",
            output_format="png",
            n=count,
        )

        results: List[bytes] = []
        for item in response.data or []:
            b64 = getattr(item, "b64_json", None)
            if not b64:
                continue
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            restored = _restore_original_dimensions(img, original_size, crop_box)
            out = io.BytesIO()
            restored.save(out, format="PNG")
            results.append(out.getvalue())

        logger.info("✅ Generated %d variations for %s", len(results), filename)
        return results or [clean_image_bytes]

    except Exception:
        logger.exception("❌ Variation generation failed: %s", filename)
        return [clean_image_bytes]
