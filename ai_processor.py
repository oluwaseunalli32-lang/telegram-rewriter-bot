import os
import re
import io
import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from PIL import Image, ImageSequence


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

WATERMARK_TEXT = "cappersfree"

# OCR enlargement.
OCR_SCALE = 2.0

# OCR confidence.
OCR_MIN_CONFIDENCE = 35

# GIF temporal detection.
TEMPORAL_STD_THRESHOLD = 8.0
TEMPORAL_DIFF_THRESHOLD = 7.0

# Expand detected watermark region slightly.
MASK_DILATION_PIXELS = 7

# OpenCV inpainting radius.
INPAINT_RADIUS = 5

# Safety limit.
# If detector thinks more than 12% of the image is watermark,
# do not modify the image.
MAX_MASK_AREA_RATIO = 0.12


# ============================================================
# CAPTION PROCESSING
# ============================================================

def replace_username(text: str) -> str:
    """
    EXISTING CAPTION BEHAVIOR.

    Only:
        1. remove *
        2. replace OLD_MENTION with NEW_MENTION

    No AI.
    No paraphrasing.
    No other caption changes.
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

    This does NOT call OpenAI.
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
# MEDIA HELPERS
# ============================================================

def _is_mp4_or_video(data: bytes) -> bool:
    """
    Detect MP4/MOV/WebM containers.

    Telegram commonly delivers GIFs as MP4.
    """

    if len(data) >= 12 and data[4:8] == b"ftyp":
        return True

    # WebM / Matroska.
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return True

    return False


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB")

    return cv2.cvtColor(
        np.asarray(rgb),
        cv2.COLOR_RGB2BGR,
    )


def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(rgb)


def _resize_for_ocr(
    image: np.ndarray,
) -> np.ndarray:

    if OCR_SCALE <= 1:
        return image

    return cv2.resize(
        image,
        None,
        fx=OCR_SCALE,
        fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC,
    )


# ============================================================
# OCR DETECTION
# ============================================================

def detect_cappersfree_text(
    image: np.ndarray,
) -> np.ndarray | None:
    """
    Detect literal @cappersfree / cappersfree text.

    Returns:
        OpenCV binary mask
        or None
    """

    try:
        enlarged = _resize_for_ocr(image)

        rgb = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2RGB,
        )

        data = pytesseract.image_to_data(
            rgb,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )

        mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        found = False

        count = len(data["text"])

        for i in range(count):

            text = (
                data["text"][i]
                or ""
            ).strip()

            if not text:
                continue

            try:
                confidence = float(
                    data["conf"][i]
                )
            except Exception:
                confidence = 0

            if confidence < OCR_MIN_CONFIDENCE:
                continue

            normalized = re.sub(
                r"[^a-z0-9@]",
                "",
                text.lower(),
            )

            if (
                WATERMARK_TEXT not in normalized
                and "@cappersfree" not in normalized
            ):
                continue

            x = int(
                data["left"][i]
                / OCR_SCALE
            )

            y = int(
                data["top"][i]
                / OCR_SCALE
            )

            w = int(
                data["width"][i]
                / OCR_SCALE
            )

            h = int(
                data["height"][i]
                / OCR_SCALE
            )

            if w <= 0 or h <= 0:
                continue

            x1 = max(0, x)
            y1 = max(0, y)

            x2 = min(
                image.shape[1],
                x + w,
            )

            y2 = min(
                image.shape[0],
                y + h,
            )

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(
                mask,
                (x1, y1),
                (x2, y2),
                255,
                -1,
            )

            found = True

        if not found:
            return None

        return _expand_mask(mask)

    except Exception:
        logger.exception(
            "⚠️ OCR watermark detection failed."
        )

        return None


# ============================================================
# MASK EXPANSION
# ============================================================

def _expand_mask(
    mask: np.ndarray | None,
    pixels: int = MASK_DILATION_PIXELS,
) -> np.ndarray | None:

    if mask is None:
        return None

    if pixels <= 0:
        return mask

    kernel_size = (
        pixels * 2
        + 1
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    return cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )


# ============================================================
# GIF TEMPORAL MASK LEARNING
# ============================================================

def learn_fixed_pulsing_mask(
    frames: list[np.ndarray],
) -> np.ndarray | None:
    """
    Learn a fixed CF watermark location from a GIF.

    Your CF graphic is expected to remain in the same location
    while pulsing/fading in and out.

    The detector compares the frames and searches for a small
    region with persistent temporal variation.

    If the result is not reliable, None is returned.
    """

    if not frames:
        return None

    if len(frames) < 3:
        return None

    try:

        height, width = frames[0].shape[:2]

        normalized = []

        for frame in frames:

            if frame.shape[:2] != (
                height,
                width,
            ):
                frame = cv2.resize(
                    frame,
                    (
                        width,
                        height,
                    ),
                    interpolation=cv2.INTER_AREA,
                )

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            normalized.append(
                gray.astype(np.float32)
            )

        stack = np.stack(
            normalized,
            axis=0,
        )

        # --------------------------------------------------------
        # Temporal standard deviation.
        # --------------------------------------------------------

        temporal_std = np.std(
            stack,
            axis=0,
        )

        # --------------------------------------------------------
        # Frame-to-frame differences.
        # --------------------------------------------------------

        diffs = np.abs(
            stack[1:]
            - stack[:-1]
        )

        temporal_diff = np.mean(
            diffs,
            axis=0,
        )

        # --------------------------------------------------------
        # Normalize.
        # --------------------------------------------------------

        std_norm = cv2.normalize(
            temporal_std,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        diff_norm = cv2.normalize(
            temporal_diff,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)

        # --------------------------------------------------------
        # Threshold.
        # --------------------------------------------------------

        std_mask = cv2.threshold(
            std_norm,
            TEMPORAL_STD_THRESHOLD,
            255,
            cv2.THRESH_BINARY,
        )[1]

        diff_mask = cv2.threshold(
            diff_norm,
            TEMPORAL_DIFF_THRESHOLD,
            255,
            cv2.THRESH_BINARY,
        )[1]

        combined = cv2.bitwise_and(
            std_mask,
            diff_mask,
        )

        # --------------------------------------------------------
        # Remove small noise.
        # --------------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_OPEN,
            kernel,
        )

        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # --------------------------------------------------------
        # Find regions.
        # --------------------------------------------------------

        contours, _ = cv2.findContours(
            combined,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        image_area = (
            width
            * height
        )

        candidates = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < 20:
                continue

            ratio = (
                area
                / image_area
            )

            if ratio > MAX_MASK_AREA_RATIO:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w < 5 or h < 5:
                continue

            roi_std = temporal_std[
                y:y + h,
                x:x + w,
            ]

            roi_diff = temporal_diff[
                y:y + h,
                x:x + w,
            ]

            if roi_std.size == 0:
                continue

            score = (
                float(np.mean(roi_std))
                +
                float(np.mean(roi_diff))
            )

            candidates.append(
                (
                    score,
                    contour,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # A CF logo can contain multiple disconnected
        # pieces: icon + letters.
        selected = candidates[:8]

        mask = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        total_area = 0

        for _, contour in selected:

            area = cv2.contourArea(
                contour
            )

            if (
                total_area
                + area
                >
                image_area
                * MAX_MASK_AREA_RATIO
            ):
                continue

            cv2.drawContours(
                mask,
                [contour],
                -1,
                255,
                -1,
            )

            total_area += area

        if cv2.countNonZero(mask) == 0:
            return None

        # --------------------------------------------------------
        # Connect pieces.
        # --------------------------------------------------------

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

        mask = _expand_mask(
            mask
        )

        final_ratio = (
            cv2.countNonZero(mask)
            / image_area
        )

        if final_ratio > MAX_MASK_AREA_RATIO:

            logger.warning(
                "⚠️ Learned mask too large: %.2f%%",
                final_ratio * 100,
            )

            return None

        logger.info(
            "🧠 Learned fixed pulsing CF mask: %.2f%% of frame",
            final_ratio * 100,
        )

        return mask

    except Exception:

        logger.exception(
            "❌ Could not learn pulsing watermark mask."
        )

        return None


# ============================================================
# MASK COMBINATION
# ============================================================

def combine_masks(
    first: np.ndarray | None,
    second: np.ndarray | None,
) -> np.ndarray | None:

    if first is None and second is None:
        return None

    if first is None:
        return second

    if second is None:
        return first

    return cv2.bitwise_or(
        first,
        second,
    )


# ============================================================
# INPAINTING
# ============================================================

def remove_watermark(
    frame: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:

    if mask is None:
        return frame

    if cv2.countNonZero(mask) == 0:
        return frame

    try:

        return cv2.inpaint(
            frame,
            mask,
            INPAINT_RADIUS,
            cv2.INPAINT_TELEA,
        )

    except Exception:

        logger.exception(
            "⚠️ Inpainting failed."
        )

        return frame


# ============================================================
# STATIC IMAGE
# ============================================================

def process_static_image(
    image_bytes: bytes,
):
    """
    Returns:

        (cleaned_bytes, "photo")

    If @cappersfree is not reliably detected,
    the original image bytes are returned unchanged.
    """

    try:

        source = Image.open(
            io.BytesIO(image_bytes)
        )

        source.load()

        original_format = (
            source.format
            or "PNG"
        ).upper()

        frame = _pil_to_bgr(
            source
        )

        # --------------------------------------------------------
        # OCR.
        # --------------------------------------------------------

        mask = detect_cappersfree_text(
            frame
        )

        if mask is None:

            logger.info(
                "ℹ️ No @cappersfree watermark detected."
            )

            # IMPORTANT:
            # Return original bytes untouched.
            return (
                image_bytes,
                "photo",
            )

        logger.info(
            "🔎 @cappersfree watermark detected."
        )

        cleaned = remove_watermark(
            frame,
            mask,
        )

        output = _bgr_to_pil(
            cleaned
        )

        buffer = io.BytesIO()

        # Preserve common image format where practical.
        if original_format in {
            "JPEG",
            "JPG",
        }:

            output.save(
                buffer,
                format="JPEG",
                quality=95,
                optimize=False,
            )

        elif original_format == "WEBP":

            output.save(
                buffer,
                format="WEBP",
                quality=95,
            )

        else:

            output.save(
                buffer,
                format="PNG",
            )

        cleaned_bytes = buffer.getvalue()

        logger.info(
            "✅ Static image cleaned: %d bytes",
            len(cleaned_bytes),
        )

        return (
            cleaned_bytes,
            "photo",
        )

    except Exception:

        logger.exception(
            "❌ STATIC IMAGE PROCESSING FAILED"
        )

        return None


# ============================================================
# ACTUAL GIF
# ============================================================

def process_gif(
    image_bytes: bytes,
):
    """
    Process an actual GIF.

    The same learned mask is applied to every frame because
    the CF logo stays in one position while pulsing.
    """

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

        for frame in ImageSequence.Iterator(
            source
        ):

            converted = frame.convert(
                "RGB"
            )

            frames.append(
                _pil_to_bgr(
                    converted
                )
            )

            durations.append(
                int(
                    frame.info.get(
                        "duration",
                        source.info.get(
                            "duration",
                            100,
                        ),
                    )
                )
            )

        if not frames:

            logger.error(
                "❌ GIF contains no frames."
            )

            return None

        logger.info(
            "🎞️ GIF contains %d frames.",
            len(frames),
        )

        # --------------------------------------------------------
        # Learn pulsing logo.
        # --------------------------------------------------------

        learned_mask = (
            learn_fixed_pulsing_mask(
                frames
            )
        )

        # --------------------------------------------------------
        # OCR across several frames.
        # --------------------------------------------------------

        ocr_mask = None

        sample_indexes = np.linspace(
            0,
            len(frames) - 1,
            min(8, len(frames)),
            dtype=int,
        )

        for index in sample_indexes:

            candidate = (
                detect_cappersfree_text(
                    frames[index]
                )
            )

            ocr_mask = combine_masks(
                ocr_mask,
                candidate,
            )

        final_mask = combine_masks(
            learned_mask,
            ocr_mask,
        )

        # --------------------------------------------------------
        # No reliable watermark.
        # --------------------------------------------------------

        if final_mask is None:

            logger.info(
                "ℹ️ No reliable CF watermark detected in GIF."
            )

            return (
                image_bytes,
                "gif",
            )

        # --------------------------------------------------------
        # Clean every frame.
        # --------------------------------------------------------

        cleaned_frames = []

        for index, frame in enumerate(
            frames
        ):

            cleaned = remove_watermark(
                frame,
                final_mask,
            )

            cleaned_frames.append(
                _bgr_to_pil(
                    cleaned
                )
            )

            logger.debug(
                "🧹 GIF frame %d/%d cleaned.",
                index + 1,
                len(frames),
            )

        # --------------------------------------------------------
        # Rebuild GIF.
        # --------------------------------------------------------

        output = io.BytesIO()

        cleaned_frames[0].save(
            output,
            format="GIF",
            save_all=True,
            append_images=cleaned_frames[1:],
            duration=durations,
            loop=loop,
            optimize=False,
        )

        cleaned_bytes = output.getvalue()

        logger.info(
            "✅ Cleaned GIF generated: %d bytes",
            len(cleaned_bytes),
        )

        return (
            cleaned_bytes,
            "gif",
        )

    except Exception:

        logger.exception(
            "❌ GIF PROCESSING FAILED"
        )

        return None


# ============================================================
# MP4 FRAME EXTRACTION
# ============================================================

def _extract_mp4_frames(
    video_bytes: bytes,
):
    """
    Extract every frame from Telegram's MP4 GIF representation.
    """

    try:

        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()

    except Exception:

        logger.exception(
            "❌ imageio-ffmpeg is unavailable."
        )

        return None

    try:

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = Path(tmp)

            input_path = (
                tmp_path
                / "input.mp4"
            )

            output_pattern = str(
                tmp_path
                / "frame_%06d.png"
            )

            input_path.write_bytes(
                video_bytes
            )

            command = [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vsync",
                "0",
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
                    "❌ FFmpeg failed: %s",
                    result.stderr.decode(
                        "utf-8",
                        errors="replace",
                    )[-3000:],
                )

                return None

            frame_paths = sorted(
                tmp_path.glob(
                    "frame_*.png"
                )
            )

            if not frame_paths:
                return None

            frames = []

            for path in frame_paths:

                with Image.open(path) as image:

                    frames.append(
                        _pil_to_bgr(
                            image.convert(
                                "RGB"
                            )
                        )
                    )

            # ----------------------------------------------------
            # Detect FPS.
            # ----------------------------------------------------

            probe_command = [
                ffmpeg,
                "-i",
                str(input_path),
            ]

            probe = subprocess.run(
                probe_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )

            stderr = probe.stderr.decode(
                "utf-8",
                errors="replace",
            )

            fps = 15.0

            match = re.search(
                r"(\d+(?:\.\d+)?)\s+fps",
                stderr,
            )

            if match:

                try:
                    fps = float(
                        match.group(1)
                    )
                except Exception:
                    pass

            duration_ms = max(
                20,
                int(
                    1000 / fps
                ),
            )

            durations = [
                duration_ms
            ] * len(frames)

            return (
                frames,
                durations,
            )

    except Exception:

        logger.exception(
            "❌ MP4 frame extraction failed."
        )

        return None


# ============================================================
# MP4 GIF PROCESSING
# ============================================================

def process_mp4_gif(
    video_bytes: bytes,
):
    """
    Telegram often sends GIFs as MP4.

    Detect the fixed pulsing CF graphic across the extracted
    frames, clean every frame, then rebuild an MP4.

    Returning MP4 means Telegram can display it as an animated
    video/GIF-style media while keeping the animation.
    """

    extracted = _extract_mp4_frames(
        video_bytes
    )

    if not extracted:
        return None

    frames, durations = extracted

    if not frames:
        return None

    logger.info(
        "🎞️ MP4/GIF contains %d frames.",
        len(frames),
    )

    learned_mask = (
        learn_fixed_pulsing_mask(
            frames
        )
    )

    # OCR.
    ocr_mask = None

    sample_indexes = np.linspace(
        0,
        len(frames) - 1,
        min(8, len(frames)),
        dtype=int,
    )

    for index in sample_indexes:

        candidate = (
            detect_cappersfree_text(
                frames[index]
            )
        )

        ocr_mask = combine_masks(
            ocr_mask,
            candidate,
        )

    final_mask = combine_masks(
        learned_mask,
        ocr_mask,
    )

    # --------------------------------------------------------
    # Nothing reliable detected.
    # --------------------------------------------------------

    if final_mask is None:

        logger.info(
            "ℹ️ No reliable CF watermark found in MP4/GIF."
        )

        # Return original MP4 untouched.
        return (
            video_bytes,
            "video",
        )

    # --------------------------------------------------------
    # Clean every frame.
    # --------------------------------------------------------

    cleaned_frames = []

    for frame in frames:

        cleaned = remove_watermark(
            frame,
            final_mask,
        )

        cleaned_frames.append(
            _bgr_to_pil(
                cleaned
            )
        )

    # --------------------------------------------------------
    # Rebuild MP4.
    # --------------------------------------------------------

    try:

        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()

    except Exception:

        logger.exception(
            "❌ Could not load FFmpeg."
        )

        return None

    try:

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = Path(tmp)

            frame_pattern = (
                tmp_path
                / "clean_%06d.png"
            )

            output_path = (
                tmp_path
                / "cleaned.mp4"
            )

            for index, frame in enumerate(
                cleaned_frames,
                start=1,
            ):

                frame.save(
                    tmp_path
                    / f"clean_{index:06d}.png",
                    format="PNG",
                )

            # Use the original-ish frame rate.
            fps = 15

            if durations:
                average_duration = (
                    sum(durations)
                    / len(durations)
                )

                if average_duration > 0:
                    fps = max(
                        1,
                        min(
                            60,
                            round(
                                1000
                                / average_duration
                            ),
                        ),
                    )

            command = [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frame_pattern),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )

            if result.returncode != 0:

                logger.error(
                    "❌ FFmpeg MP4 rebuild failed: %s",
                    result.stderr.decode(
                        "utf-8",
                        errors="replace",
                    )[-3000:],
                )

                return None

            cleaned_bytes = (
                output_path.read_bytes()
            )

            logger.info(
                "✅ Cleaned MP4 generated: %d bytes",
                len(cleaned_bytes),
            )

            return (
                cleaned_bytes,
                "video",
            )

    except Exception:

        logger.exception(
            "❌ MP4 rebuild failed."
        )

        return None


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

async def remove_watermarks_from_bytes(
    image_bytes: bytes,
):
    """
    MAIN FUNCTION USED BY main.py.

    Pipeline:

        ORIGINAL
            ↓
        DETECT CF
            ↓
        MASK
            ↓
        INPAINT
            ↓
        CLEANED ORIGINAL

    There is NO:
        OpenAI
        Vision
        image generation
        reconstruction

    Returns:

        (bytes, "photo")
        (bytes, "gif")
        (bytes, "video")

    or:

        None
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
        "🧹 WATERMARK REMOVAL PIPELINE START"
    )

    logger.info(
        "📦 Original bytes: %d",
        len(image_bytes),
    )

    logger.info(
        "🔬 Signature: %s",
        bytes(
            image_bytes[:16]
        ).hex(" "),
    )

    try:

        # --------------------------------------------------------
        # Actual GIF.
        # --------------------------------------------------------

        if (
            image_bytes.startswith(
                b"GIF87a"
            )
            or
            image_bytes.startswith(
                b"GIF89a"
            )
        ):

            logger.info(
                "🎞️ Actual GIF detected."
            )

            return process_gif(
                image_bytes
            )

        # --------------------------------------------------------
        # MP4 / MOV / WebM.
        # --------------------------------------------------------

        if _is_mp4_or_video(
            image_bytes
        ):

            logger.info(
                "🎞️ MP4/video container detected."
            )

            return process_mp4_gif(
                image_bytes
            )

        # --------------------------------------------------------
        # Static image.
        # --------------------------------------------------------

        logger.info(
            "🖼️ Static image detected."
        )

        return process_static_image(
            image_bytes
        )

    except Exception:

        logger.exception(
            "❌ WATERMARK REMOVAL PIPELINE FAILED"
        )

        return None

    finally:

        logger.info(
            "🧹 WATERMARK REMOVAL PIPELINE END"
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
    """
    Compatibility alias.

    IMPORTANT:
    This does NOT regenerate an image.

    It now performs watermark removal only.

    Kept so an older main.py cannot accidentally break the
    application because of the old function name.
    """

    return await remove_watermarks_from_bytes(
        image_bytes
    )
