import os
import io
import re
import base64
import logging
from typing import Optional, Tuple

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
OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "low").strip()

# Beta safety: at most ONE OpenAI still-image edit per process run.
BETA_MODE = os.getenv("BETA_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
MAX_OPENAI_EDITS = int(os.getenv("MAX_OPENAI_EDITS_PER_RUN", "1"))

NEW_MENTION = os.getenv("NEW_MENTION", "@PrimeAnalysiss").strip()
if NEW_MENTION and not NEW_MENTION.startswith("@"):
    NEW_MENTION = "@" + NEW_MENTION

OLD_MENTION = "@cappersfree"

_openai_edits_used = 0
client = AsyncOpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and AsyncOpenAI) else None


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
# LOCAL DETECTION HELPERS
# ============================================================

def _bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def detect_red_watermark_mask_bgr(image: np.ndarray) -> np.ndarray:
    """Search the entire image for saturated red watermark-like pixels."""
    hsv = _bgr_to_hsv(image)

    # Pure/strong red, including slightly softened anti-aliased edges.
    lower1 = np.array([0, 100, 90], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 100, 90], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # Join broken glyph strokes without swallowing large areas.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    # Keep only components that look like text/branding rather than huge red UI blocks.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    h, w = mask.shape[:2]

    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < max(10, int(w * h * 0.00001)):
            continue
        if area > w * h * 0.03:
            continue

        aspect = cw / max(ch, 1)
        # Text-like or compact logo-like components.
        if aspect >= 1.4 or (cw >= 12 and ch >= 12):
            out[labels == i] = 255

    return out


def detect_green_logo_mask_bgr(image: np.ndarray) -> np.ndarray:
    """Detect the CF-style green circular logo without assuming one fixed position.

    We search the whole frame for compact green components, score them for
    circularity, and slightly prefer the upper-right area observed in the
    supplied beta GIF/MP4. This avoids accidentally selecting the green
    checkmark in the message content.
    """
    hsv = _bgr_to_hsv(image)
    green = cv2.inRange(
        hsv,
        np.array([35, 140, 80], dtype=np.uint8),
        np.array([90, 255, 255], dtype=np.uint8),
    )

    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(green)

    h, w = green.shape[:2]
    best = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 3 or ch < 3:
            continue

        bbox_area = float(cw * ch)
        fill = area / bbox_area if bbox_area else 0.0
        perimeter = cv2.arcLength(contour, True)
        circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0

        if not (0.55 <= cw / max(ch, 1) <= 1.8):
            continue

        # The CF ring is relatively circular and not a solid rectangle.
        if circularity < 0.35:
            continue

        cx = x + cw / 2.0
        cy = y + ch / 2.0

        # Prefer, but do not require, the upper-right quadrant where the supplied
        # animated sample places the CF logo.
        tx = w * 0.92
        ty = h * 0.28
        dist = np.hypot((cx - tx) / max(w, 1), (cy - ty) / max(h, 1))
        location_bonus = max(0.0, 1.0 - dist * 2.5)

        score = (area ** 0.55) * (0.5 + circularity) * (0.7 + location_bonus)
        candidate = (score, x, y, cw, ch, cx, cy, area)

        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return np.zeros_like(green)

    _, x, y, cw, ch, cx, cy, _ = best

    # Recover the whole circular logo, including black/white interior.
    radius = max(cw, ch) * 0.72
    radius = max(radius, 8.0)

    out = np.zeros_like(green)
    cv2.circle(
        out,
        (int(round(cx)), int(round(cy))),
        int(round(radius)),
        255,
        -1,
    )

    return out

def combine_masks(*masks: np.ndarray, dilation: int = 2) -> np.ndarray:
    valid = [m for m in masks if m is not None]
    if not valid:
        raise ValueError("No masks provided")

    out = np.zeros_like(valid[0])
    for m in valid:
        out = cv2.bitwise_or(out, m)

    if dilation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation * 2 + 1, dilation * 2 + 1),
        )
        out = cv2.dilate(out, kernel, iterations=1)

    return out


def mask_has_content(mask: np.ndarray) -> bool:
    pixels = int(cv2.countNonZero(mask))
    total = mask.shape[0] * mask.shape[1]
    if total <= 0:
        return False

    ratio = pixels / total
    logger.info("🔎 Local watermark mask coverage: %.4f%%", ratio * 100)

    # Conservative guard. Never call the API for a huge mask.
    return 0.00003 <= ratio <= 0.08


# ============================================================
# STILL IMAGE -> OPENAI EDIT
# ============================================================

def _prepare_still(image_bytes: bytes) -> Tuple[Image.Image, np.ndarray]:
    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    max_dimension = 1536
    if max(source.size) > max_dimension:
        scale = max_dimension / max(source.size)
        source = source.resize(
            (
                max(1, int(source.width * scale)),
                max(1, int(source.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )

    bgr = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
    red_mask = detect_red_watermark_mask_bgr(bgr)
    green_mask = detect_green_logo_mask_bgr(bgr)
    combined = combine_masks(red_mask, green_mask, dilation=2)

    return source, combined


async def remove_watermarks_from_bytes(
    image_bytes: bytes,
    filename: str = "image.jpg",
):
    """Beta still-image cleanup. One OpenAI edit max per process run."""
    global _openai_edits_used

    if not image_bytes:
        return None

    # Strong safety switch.
    if BETA_MODE and _openai_edits_used >= MAX_OPENAI_EDITS:
        logger.warning(
            "🛑 BETA AI LIMIT reached (%d). Keeping original: %s",
            MAX_OPENAI_EDITS,
            filename,
        )
        return None

    if not client:
        logger.warning("⚠️ OPENAI_API_KEY missing; keeping original.")
        return None

    try:
        source, mask = _prepare_still(image_bytes)

        if not mask_has_content(mask):
            logger.info("✅ No strong watermark candidate found: %s", filename)
            return None

        image_buffer = io.BytesIO()
        source.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        image_buffer.name = "input.png"

        mask_buffer = io.BytesIO()
        # White = area to modify, black = preserve.
        Image.fromarray(mask).save(mask_buffer, format="PNG")
        mask_buffer.seek(0)
        mask_buffer.name = "mask.png"

        logger.warning(
            "💰 BETA: using OpenAI edit %d/%d for %s | model=%s | quality=%s",
            _openai_edits_used + 1,
            MAX_OPENAI_EDITS,
            filename,
            OPENAI_IMAGE_MODEL,
            OPENAI_IMAGE_QUALITY,
        )

        _openai_edits_used += 1

        response = await client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=image_buffer,
            mask=mask_buffer,
            prompt=(
                "Remove only the detected Cappersfree watermark elements in the masked areas. "
                "Reconstruct the underlying image naturally. Preserve everything else exactly: "
                "subjects, faces, UI text, odds, scores, layout, colors, lighting, borders, "
                "and composition. Do not crop, resize, redesign, or invent content outside "
                "the masked regions. The red @cappersfree watermark and CF logo are unwanted "
                "branding and must be removed completely."
            ),
            quality=OPENAI_IMAGE_QUALITY,
            output_format="png",
        )

        if not response.data:
            logger.error("❌ OpenAI returned no image data.")
            return None

        b64 = getattr(response.data[0], "b64_json", None)
        if not b64:
            logger.error("❌ OpenAI response did not include b64_json.")
            return None

        cleaned = base64.b64decode(b64)
        if not cleaned:
            logger.error("❌ OpenAI returned an empty result.")
            return None

        logger.info("✅ Beta still-image edit completed: %s", filename)
        return cleaned

    except Exception:
        # IMPORTANT: request count has already been consumed once the call is made.
        logger.exception("❌ Beta still-image processing failed: %s", filename)
        return None


async def regenerate_image_from_bytes(image_bytes: bytes, filename: str = "image.jpg"):
    return await remove_watermarks_from_bytes(image_bytes, filename)


# ============================================================
# GIF / MP4 LOCAL PROCESSING
# ============================================================

def _inpaint_frame(frame: np.ndarray) -> np.ndarray:
    red = detect_red_watermark_mask_bgr(frame)
    green = detect_green_logo_mask_bgr(frame)
    mask = combine_masks(red, green, dilation=3)

    if not mask_has_content(mask):
        return frame

    # Telea is fast and appropriate for a beta pass on small overlays.
    return cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)


def _detect_cf_logo_circle(frame: np.ndarray):
    """Detect the animated CF circle in the supplied beta GIF/MP4 sample.

    For this watermark family the logo is in the upper-right portion of the
    animation. We intentionally avoid generic green UI elements elsewhere in
    the message (for example, the green check icon).
    """
    hsv = _bgr_to_hsv(frame)
    green = cv2.inRange(
        hsv,
        np.array([35, 120, 60], dtype=np.uint8),
        np.array([95, 255, 255], dtype=np.uint8),
    )

    h, w = green.shape[:2]
    x_start = int(w * 0.74)
    y_end = int(h * 0.70)
    roi = green[:y_end, x_start:]

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (roi > 0).astype(np.uint8),
        8,
    )

    best = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 8 or cw < 3 or ch < 3:
            continue

        # The supplied CF logo's green ring is a compact component.
        # Prefer the largest component in the restricted area.
        if best is None or area > best[0]:
            best = (area, i, x, y, cw, ch)

    if best is None:
        return None

    _, idx, _, _, _, _ = best
    yy, xx = np.where(labels == idx)
    if len(xx) < 3:
        return None

    xx = xx + x_start
    pts = np.column_stack([xx, yy]).astype(np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(pts)

    # Expand to cover the black/white interior of the complete logo.
    radius = max(radius + 8.0, 12.0)

    mask = np.zeros_like(green)
    cv2.circle(
        mask,
        (int(round(cx)), int(round(cy))),
        int(round(radius)),
        255,
        -1,
    )
    return mask

def remove_watermarks_from_video_bytes(video_bytes: bytes, filename: str = "video.mp4") -> Optional[bytes]:
    """Beta local frame-by-frame cleanup for MP4/video. No OpenAI calls."""
    temp_in = None
    temp_video = None
    temp_final = None

    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(video_bytes)
            temp_in = f.name

        cap = cv2.VideoCapture(temp_in)
        if not cap.isOpened():
            logger.error("❌ Could not open video: %s", filename)
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            temp_video = f.name

        writer = cv2.VideoWriter(
            temp_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            logger.error("❌ Could not create output video: %s", filename)
            return None

        changed = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Different masks use different inpaint radii. The red text is thin;
            # the animated CF logo is large and needs a much larger reconstruction radius.
            red = detect_red_watermark_mask_bgr(frame)
            if cv2.countNonZero(red):
                frame = cv2.inpaint(frame, red, 4, cv2.INPAINT_TELEA)
                changed += 1

            logo = _detect_cf_logo_circle(frame)
            if logo is not None and cv2.countNonZero(logo):
                frame = cv2.inpaint(frame, logo, 25, cv2.INPAINT_TELEA)
                changed += 1

            writer.write(frame)

        cap.release()
        writer.release()

        if not os.path.exists(temp_video):
            return None

        # Reattach source audio when ffmpeg is available. The beta sample is
        # silent, but this keeps real MP4 audio from being lost.
        import subprocess
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            temp_final = f.name

        ffmpeg_ok = False
        try:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", temp_video,
                "-i", temp_in,
                "-map", "0:v:0",
                "-map", "1:a?",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "18",
                "-c:a", "copy",
                "-shortest",
                temp_final,
            ]
            subprocess.run(cmd, check=True)
            ffmpeg_ok = os.path.exists(temp_final) and os.path.getsize(temp_final) > 0
        except Exception:
            logger.warning("⚠️ ffmpeg remux unavailable; returning processed video without original audio.")

        result_path = temp_final if ffmpeg_ok else temp_video
        with open(result_path, "rb") as f:
            result = f.read()

        logger.info(
            "🎞️ Beta video cleanup complete: %d source frames | output=%d bytes | audio_remux=%s",
            frame_count, len(result), ffmpeg_ok,
        )
        return result

    except Exception:
        logger.exception("❌ Beta video processing failed: %s", filename)
        return None

    finally:
        for p in (temp_in, temp_video, temp_final):
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass


def remove_watermarks_from_gif_bytes(gif_bytes: bytes, filename: str = "animation.gif") -> Optional[bytes]:
    """Beta local GIF cleanup while preserving GIF timing/loop metadata."""
    try:
        source = Image.open(io.BytesIO(gif_bytes))
        frames = []
        durations = []
        loop = source.info.get("loop", 0)

        for index in range(getattr(source, "n_frames", 1)):
            source.seek(index)
            rgba = source.convert("RGBA")
            bgr = cv2.cvtColor(np.array(rgba), cv2.COLOR_RGBA2BGR)

            red = detect_red_watermark_mask_bgr(bgr)
            if cv2.countNonZero(red):
                bgr = cv2.inpaint(bgr, red, 4, cv2.INPAINT_TELEA)

            logo = _detect_cf_logo_circle(bgr)
            if logo is not None and cv2.countNonZero(logo):
                bgr = cv2.inpaint(bgr, logo, 25, cv2.INPAINT_TELEA)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb).convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
            durations.append(source.info.get("duration", 100))

        if not frames:
            return None

        out = io.BytesIO()
        frames[0].save(
            out,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=loop,
            disposal=2,
        )
        return out.getvalue()

    except Exception:
        logger.exception("❌ Beta GIF processing failed: %s", filename)
        return None

