import os
import re
import io
import cv2
import time
import base64
import logging
import tempfile
import subprocess

import numpy as np

from pathlib import Path
from PIL import Image, ImageSequence
from aiogram.types import BufferedInputFile


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("ai_processor")


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
# WATERMARK CONFIGURATION
# ============================================================

# Red watermark detection.
#
# The @cappersfree text in the supplied graphic is red, so we
# specifically look for red pixels rather than modifying the
# entire image.
#
# These values are intentionally reasonably broad because
# Telegram compression can change the exact red color.

RED_H_LOW_1 = 0
RED_H_HIGH_1 = 12

RED_H_LOW_2 = 170
RED_H_HIGH_2 = 179

RED_S_MIN = 80
RED_V_MIN = 70


# Minimum connected component size to consider.
MIN_RED_COMPONENT_AREA = 8


# ============================================================
# CAPTION PROCESSING
# ============================================================

def replace_username(text: str) -> str:
    """
    Caption behavior is intentionally unchanged.

    1. Remove literal *
    2. Replace OLD_MENTION with NEW_MENTION

    Nothing else is changed.
    """

    if not text:
        return text

    result = text.replace("*", "")

    if NEW_MENTION:
        result = re.sub(
            re.escape(OLD_MENTION),
            NEW_MENTION,
            result,
            flags=re.IGNORECASE,
        )
    else:
        logger.warning(
            "NEW_MENTION is empty. Username was not replaced."
        )

    return result


async def rewrite_text(original_text: str) -> str:
    """
    Kept for compatibility with main.py.
    """

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
# IMAGE HELPERS
# ============================================================

def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")
    return cv2.cvtColor(
        np.array(rgb),
        cv2.COLOR_RGB2BGR,
    )


def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(rgb)


def _resize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean and expand the detected watermark mask slightly.

    Expansion is important because anti-aliased edges of
    @cappersfree can be partially red rather than fully red.
    """

    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    kernel_expand = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_small,
    )

    mask = cv2.dilate(
        mask,
        kernel_expand,
        iterations=1,
    )

    return mask


# ============================================================
# RED @CAPPERSFREE DETECTION
# ============================================================

def detect_red_watermark_mask(frame: np.ndarray) -> np.ndarray:
    """
    Detect red watermark text such as @cappersfree.

    This does NOT remove all red in the image blindly.

    Small red areas are retained for connected-component
    analysis, and only suitable watermark-like regions are
    accepted.
    """

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    lower_red_1 = np.array(
        [
            RED_H_LOW_1,
            RED_S_MIN,
            RED_V_MIN,
        ],
        dtype=np.uint8,
    )

    upper_red_1 = np.array(
        [
            RED_H_HIGH_1,
            255,
            255,
        ],
        dtype=np.uint8,
    )

    lower_red_2 = np.array(
        [
            RED_H_LOW_2,
            RED_S_MIN,
            RED_V_MIN,
        ],
        dtype=np.uint8,
    )

    upper_red_2 = np.array(
        [
            RED_H_HIGH_2,
            255,
            255,
        ],
        dtype=np.uint8,
    )

    mask1 = cv2.inRange(
        hsv,
        lower_red_1,
        upper_red_1,
    )

    mask2 = cv2.inRange(
        hsv,
        lower_red_2,
        upper_red_2,
    )

    red_mask = cv2.bitwise_or(
        mask1,
        mask2,
    )

    # Connected components.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        red_mask,
        connectivity=8,
    )

    filtered = np.zeros_like(red_mask)

    height, width = red_mask.shape[:2]

    for i in range(1, count):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_RED_COMPONENT_AREA:
            continue

        # Ignore giant regions. They are much more likely to
        # be legitimate graphic elements than watermark text.
        if area > width * height * 0.12:
            continue

        # Ignore extremely large full-height/width areas.
        if w > width * 0.60 or h > height * 0.40:
            continue

        # Watermark text is generally relatively thin.
        if w >= 10 and h >= 3:
            filtered[labels == i] = 255

    return _resize_mask(filtered)


# ============================================================
# OCR-BASED @CAPPERSFREE DETECTION
# ============================================================

def detect_cappersfree_ocr_mask(frame: np.ndarray) -> np.ndarray:
    """
    Optional OCR detection.

    pytesseract is optional. If the Tesseract executable is not
    installed, this simply returns an empty mask.

    This is useful when Telegram compression makes the red
    detector less reliable.
    """

    mask = np.zeros(
        frame.shape[:2],
        dtype=np.uint8,
    )

    try:
        import pytesseract
    except ImportError:
        return mask

    try:
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        data = pytesseract.image_to_data(
            rgb,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )

        for i, raw_text in enumerate(
            data.get("text", [])
        ):

            text = (raw_text or "").strip().lower()

            compact = re.sub(
                r"[^a-z0-9@]",
                "",
                text,
            )

            if (
                "cappersfree" in compact
                or "cappers" in compact
                or "@cappersfree" in compact
            ):

                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])

                if w > 0 and h > 0:

                    pad_x = max(
                        4,
                        int(w * 0.15),
                    )

                    pad_y = max(
                        4,
                        int(h * 0.30),
                    )

                    x1 = max(
                        0,
                        x - pad_x,
                    )

                    y1 = max(
                        0,
                        y - pad_y,
                    )

                    x2 = min(
                        frame.shape[1],
                        x + w + pad_x,
                    )

                    y2 = min(
                        frame.shape[0],
                        y + h + pad_y,
                    )

                    mask[
                        y1:y2,
                        x1:x2
                    ] = 255

    except Exception:
        logger.debug(
            "OCR watermark detection unavailable.",
            exc_info=True,
        )

    return mask


# ============================================================
# TEMPORAL CF LOGO DETECTION
# ============================================================

def detect_pulsing_overlay_mask(
    frames,
) -> np.ndarray:
    """
    Learn an animated/pulsing CF overlay from the GIF itself.

    The assumption is:

      - the actual graphic remains mostly stable
      - the CF overlay changes brightness/opacity
      - therefore pixels around the CF graphic change between
        frames while nearby legitimate content remains stable

    We compare frames and build a temporal-change map.

    This is specifically useful for a CF logo that pulses
    in/out while staying in one location.
    """

    if not frames or len(frames) < 2:
        return None

    base = frames[0]

    h, w = base.shape[:2]

    differences = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    gray_frames = []

    for frame in frames:

        if frame.shape[:2] != (h, w):
            frame = cv2.resize(
                frame,
                (w, h),
            )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        gray_frames.append(gray)

    for i in range(
        1,
        len(gray_frames),
    ):

        diff = cv2.absdiff(
            gray_frames[i],
            gray_frames[i - 1],
        )

        differences += diff.astype(
            np.float32
        )

    differences /= max(
        1,
        len(gray_frames) - 1,
    )

    # Dynamic overlay candidates.
    dynamic = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    # Only keep meaningful repeated changes.
    dynamic[differences >= 18] = 255

    # Remove isolated noise.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    dynamic = cv2.morphologyEx(
        dynamic,
        cv2.MORPH_OPEN,
        kernel,
    )

    dynamic = cv2.morphologyEx(
        dynamic,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Find connected regions.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dynamic,
        connectivity=8,
    )

    result = np.zeros_like(dynamic)

    for i in range(1, count):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 20:
            continue

        if area > w * h * 0.08:
            continue

        # Avoid treating huge portions of a changing graphic
        # as a watermark.
        if ww > w * 0.35 or hh > h * 0.35:
            continue

        result[labels == i] = 255

    if not np.any(result):
        return None

    # Slightly enlarge to cover anti-aliased edges.
    result = cv2.dilate(
        result,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7),
        ),
        iterations=1,
    )

    return result


# ============================================================
# INPAINTING
# ============================================================

def remove_mask_from_frame(
    frame: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Remove the detected watermark using OpenCV inpainting.

    Telea is generally good for text and small graphic overlays.
    """

    if mask is None:
        return frame

    if not np.any(mask):
        return frame

    # Final safety cleanup.
    mask = cv2.GaussianBlur(
        mask,
        (3, 3),
        0,
    )

    _, mask = cv2.threshold(
        mask,
        20,
        255,
        cv2.THRESH_BINARY,
    )

    return cv2.inpaint(
        frame,
        mask,
        5,
        cv2.INPAINT_TELEA,
    )


# ============================================================
# STATIC IMAGE PROCESSING
# ============================================================

def process_static_image(
    image_bytes: bytes,
) -> bytes | None:

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image.load()

        frame = _pil_to_bgr(image)

        # --------------------------------------------
        # Red @cappersfree detection
        # --------------------------------------------

        red_mask = detect_red_watermark_mask(
            frame
        )

        # --------------------------------------------
        # OCR backup
        # --------------------------------------------

        ocr_mask = detect_cappersfree_ocr_mask(
            frame
        )

        mask = cv2.bitwise_or(
            red_mask,
            ocr_mask,
        )

        red_pixels = int(
            np.count_nonzero(red_mask)
        )

        ocr_pixels = int(
            np.count_nonzero(ocr_mask)
        )

        logger.info(
            "🔴 Red watermark mask: %d pixels",
            red_pixels,
        )

        logger.info(
            "🔎 OCR watermark mask: %d pixels",
            ocr_pixels,
        )

        # ------------------------------------------------
        # IMPORTANT:
        #
        # For a static image there is no temporal information
        # with which to automatically distinguish a non-red CF
        # logo from legitimate graphics.
        #
        # Therefore we do NOT blindly erase arbitrary non-red
        # areas.
        # ------------------------------------------------

        processed = remove_mask_from_frame(
            frame,
            mask,
        )

        output = io.BytesIO()

        result = _bgr_to_pil(
            processed
        )

        # Preserve PNG where possible.
        original_format = (
            (image.format or "PNG").upper()
        )

        if original_format == "JPEG":
            result.save(
                output,
                format="JPEG",
                quality=97,
            )
        elif original_format == "WEBP":
            result.save(
                output,
                format="WEBP",
                quality=97,
            )
        else:
            result.save(
                output,
                format="PNG",
            )

        return output.getvalue()

    except Exception:
        logger.exception(
            "❌ STATIC IMAGE WATERMARK REMOVAL FAILED"
        )

        return None


# ============================================================
# GIF PROCESSING
# ============================================================

def process_gif(
    image_bytes: bytes,
) -> bytes | None:

    try:

        source = Image.open(
            io.BytesIO(image_bytes)
        )

        frames = []

        durations = []

        loop = source.info.get(
            "loop",
            0,
        )

        disposal = []

        for frame in ImageSequence.Iterator(
            source
        ):

            frame_copy = frame.convert(
                "RGB"
            )

            bgr = _pil_to_bgr(
                frame_copy
            )

            frames.append(
                bgr
            )

            durations.append(
                frame.info.get(
                    "duration",
                    source.info.get(
                        "duration",
                        100,
                    ),
                )
            )

            disposal.append(
                frame.info.get(
                    "disposal",
                    0,
                )
            )

        if not frames:
            return None

        logger.info(
            "🎞️ GIF contains %d frame(s).",
            len(frames),
        )

        # ====================================================
        # LEARN THE PULSING CF OVERLAY
        # ====================================================

        temporal_mask = detect_pulsing_overlay_mask(
            frames
        )

        if temporal_mask is not None:

            logger.info(
                "🧠 Learned temporal/pulsing overlay mask from GIF."
            )

            logger.info(
                "🧠 Temporal mask pixels: %d",
                int(
                    np.count_nonzero(
                        temporal_mask
                    )
                ),
            )

        else:

            logger.info(
                "🧠 No reliable temporal overlay detected."
            )

        processed_frames = []

        for index, frame in enumerate(
            frames
        ):

            # Red @cappersfree detection on
            # every frame.
            red_mask = detect_red_watermark_mask(
                frame
            )

            # OCR backup on every frame.
            ocr_mask = detect_cappersfree_ocr_mask(
                frame
            )

            mask = cv2.bitwise_or(
                red_mask,
                ocr_mask,
            )

            # ------------------------------------------------
            # Add learned pulsing overlay.
            #
            # This is only used when the GIF itself shows
            # repeated temporal changes.
            # ------------------------------------------------

            if temporal_mask is not None:

                mask = cv2.bitwise_or(
                    mask,
                    temporal_mask,
                )

            processed = remove_mask_from_frame(
                frame,
                mask,
            )

            processed_frames.append(
                _bgr_to_pil(
                    processed
                )
            )

            logger.info(
                "🎞️ Processed GIF frame %d/%d",
                index + 1,
                len(frames),
            )

        # ====================================================
        # WRITE NEW GIF
        # ====================================================

        output = io.BytesIO()

        processed_frames[0].save(
            output,
            format="GIF",
            save_all=True,
            append_images=processed_frames[1:],
            duration=durations,
            loop=loop,
            disposal=disposal,
            optimize=False,
        )

        logger.info(
            "✅ GIF watermark removal complete."
        )

        return output.getvalue()

    except Exception:
        logger.exception(
            "❌ GIF WATERMARK REMOVAL FAILED"
        )

        return None


# ============================================================
# MP4 / TELEGRAM GIF PROCESSING
# ============================================================

def process_video(
    video_bytes: bytes,
) -> bytes | None:

    try:

        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()

    except Exception:
        logger.exception(
            "❌ imageio-ffmpeg unavailable."
        )
        return None

    try:

        with tempfile.TemporaryDirectory() as tmp:

            input_path = (
                Path(tmp) / "input.mp4"
            )

            output_path = (
                Path(tmp) / "output.mp4"
            )

            input_path.write_bytes(
                video_bytes
            )

            # --------------------------------------------
            # Extract frames as PNG.
            # --------------------------------------------

            frames_dir = (
                Path(tmp) / "frames"
            )

            frames_dir.mkdir()

            frame_pattern = str(
                frames_dir / "frame_%06d.png"
            )

            extract_command = [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vsync",
                "0",
                frame_pattern,
            ]

            result = subprocess.run(
                extract_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )

            if result.returncode != 0:
                logger.error(
                    "❌ FFmpeg extraction failed: %s",
                    result.stderr.decode(
                        "utf-8",
                        errors="replace",
                    )[-3000:],
                )
                return None

            frame_paths = sorted(
                frames_dir.glob(
                    "frame_*.png"
                )
            )

            if not frame_paths:
                logger.error(
                    "❌ No video frames extracted."
                )
                return None

            frames = []

            for path in frame_paths:

                image = cv2.imread(
                    str(path),
                    cv2.IMREAD_COLOR,
                )

                if image is not None:
                    frames.append(
                        image
                    )

            if not frames:
                return None

            logger.info(
                "🎞️ Video/GIF MP4 contains %d frame(s).",
                len(frames),
            )

            # =================================================
            # LEARN PULSING CF MASK
            # =================================================

            temporal_mask = detect_pulsing_overlay_mask(
                frames
            )

            processed_paths = []

            for index, frame in enumerate(
                frames
            ):

                red_mask = detect_red_watermark_mask(
                    frame
                )

                ocr_mask = detect_cappersfree_ocr_mask(
                    frame
                )

                mask = cv2.bitwise_or(
                    red_mask,
                    ocr_mask,
                )

                if temporal_mask is not None:
                    mask = cv2.bitwise_or(
                        mask,
                        temporal_mask,
                    )

                processed = remove_mask_from_frame(
                    frame,
                    mask,
                )

                output_frame = (
                    frames_dir
                    / f"processed_{index:06d}.png"
                )

                cv2.imwrite(
                    str(output_frame),
                    processed,
                )

                processed_paths.append(
                    output_frame
                )

                logger.info(
                    "🎞️ Processed video frame %d/%d",
                    index + 1,
                    len(frames),
                )

            # =================================================
            # REBUILD MP4
            # =================================================

            processed_pattern = str(
                frames_dir / "processed_%06d.png"
            )

            encode_command = [
                ffmpeg,
                "-y",
                "-framerate",
                "30",
                "-i",
                processed_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

            result = subprocess.run(
                encode_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )

            if result.returncode != 0:
                logger.error(
                    "❌ FFmpeg encoding failed: %s",
                    result.stderr.decode(
                        "utf-8",
                        errors="replace",
                    )[-3000:],
                )
                return None

            if not output_path.exists():
                return None

            output_bytes = (
                output_path.read_bytes()
            )

            logger.info(
                "✅ MP4/GIF watermark removal complete: %d bytes",
                len(output_bytes),
            )

            return output_bytes

    except Exception:
        logger.exception(
            "❌ VIDEO WATERMARK REMOVAL FAILED"
        )

        return None


# ============================================================
# MEDIA TYPE DETECTION
# ============================================================

def _is_mp4(data: bytes) -> bool:

    return (
        len(data) >= 12
        and data[4:8] == b"ftyp"
    )


def _is_gif(data: bytes) -> bool:

    return data.startswith(
        b"GIF87a"
    ) or data.startswith(
        b"GIF89a"
    )


# ============================================================
# MAIN WATERMARK REMOVAL FUNCTION
# ============================================================

async def remove_watermarks_from_bytes(
    image_bytes: bytes,
    filename: str = "",
    mime_type: str = "",
):
    """
    Main entry point used by main.py.

    The image is NOT regenerated.

    The original pixels are processed directly.
    """

    if not image_bytes:
        logger.error(
            "❌ remove_watermarks_from_bytes received empty bytes."
        )
        return None

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    logger.info(
        "🧹 WATERMARK REMOVAL START"
    )

    logger.info(
        "📦 Original bytes: %d",
        len(image_bytes),
    )

    logger.info(
        "🔬 Signature: %s",
        image_bytes[:32].hex(" "),
    )

    logger.info(
        "📄 Filename: %s",
        filename,
    )

    logger.info(
        "📄 MIME: %s",
        mime_type,
    )

    # ========================================================
    # GIF
    # ========================================================

    if _is_gif(image_bytes):

        logger.info(
            "🎞️ Actual GIF detected."
        )

        result = process_gif(
            image_bytes
        )

        if result:
            return BufferedInputFile(
                result,
                filename="watermark_removed.gif",
            )

        return None

    # ========================================================
    # MP4 / Telegram GIF
    # ========================================================

    if _is_mp4(image_bytes):

        logger.info(
            "🎞️ MP4 container detected."
        )

        result = process_video(
            image_bytes
        )

        if result:

            return BufferedInputFile(
                result,
                filename="watermark_removed.mp4",
            )

        return None

    # ========================================================
    # NORMAL IMAGE
    # ========================================================

    logger.info(
        "🖼️ Static image detected."
    )

    result = process_static_image(
        image_bytes
    )

    if result:

        return BufferedInputFile(
            result,
            filename="watermark_removed.png",
        )

    return None


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

async def regenerate_image_from_bytes(
    image_bytes: bytes,
):
    """
    Compatibility wrapper.

    Old main.py versions may still call this function.

    It now performs watermark removal instead of AI
    regeneration.
    """

    return await remove_watermarks_from_bytes(
        image_bytes
    )
