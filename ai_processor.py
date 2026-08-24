import os
import re
import io
import time
import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytesseract

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

WATERMARK_TEXT = "cappersfree"

# Conservative settings.
#
# The processor tries to detect the watermark automatically.
# It does NOT use a hard-coded coordinate.

OCR_SCALE = 2.0

# Minimum confidence before OCR text is considered useful.
OCR_MIN_CONFIDENCE = 35

# Temporal detection thresholds.
TEMPORAL_STD_THRESHOLD = 8.0
TEMPORAL_DIFF_THRESHOLD = 7.0

# Morphological expansion around detected watermark.
MASK_DILATION_PIXELS = 7

# Inpainting radius.
INPAINT_RADIUS = 5

# If the automatically detected region is absurdly large,
# don't touch the image.
MAX_MASK_AREA_RATIO = 0.12


# ============================================================
# CAPTION PROCESSING
# ============================================================

def replace_username(text: str) -> str:
    """
    Caption processing remains EXACTLY as requested.

    1. Remove '*'
    2. Replace OLD_MENTION with NEW_MENTION

    No AI rewriting.
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
    Kept with the same function name so main.py
    does not need to change its caption interface.
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
# GENERAL HELPERS
# ============================================================

def _is_mp4_or_video(data: bytes) -> bool:
    """
    Telegram GIFs are frequently delivered as MP4.
    """

    if len(data) >= 12 and data[4:8] == b"ftyp":
        return True

    # WebM / Matroska
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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _resize_for_ocr(image: np.ndarray) -> np.ndarray:
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
    Try to locate literal @cappersfree text.

    Returns a mask or None.

    OCR is only one detector. The GIF temporal detector
    handles graphical/pulsing CF material.
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

            # Direct detection.
            matches = (
                "cappersfree" in normalized
                or "@cappersfree" in normalized
            )

            if not matches:
                continue

            x = int(data["left"][i] / OCR_SCALE)
            y = int(data["top"][i] / OCR_SCALE)
            w = int(data["width"][i] / OCR_SCALE)
            h = int(data["height"][i] / OCR_SCALE)

            if w <= 0 or h <= 0:
                continue

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(image.shape[1], x + w)
            y2 = min(image.shape[0], y + h)

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
    mask: np.ndarray,
    pixels: int = MASK_DILATION_PIXELS,
) -> np.ndarray:

    if mask is None:
        return None

    if pixels <= 0:
        return mask

    kernel_size = pixels * 2 + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
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
    Automatically learn a fixed watermark region from GIF frames.

    Assumption for this application:

        CF logo remains in one location
        but pulses/fades in and out.

    Therefore the logo produces persistent temporal variation
    in approximately the same spatial region.

    The detector:

        1. Converts frames to grayscale.
        2. Computes temporal standard deviation.
        3. Computes frame-to-frame differences.
        4. Combines the two signals.
        5. Finds persistent connected regions.
        6. Rejects enormous regions.
        7. Expands the resulting region.

    It intentionally avoids changing the image when confidence
    is too low.
    """

    if not frames or len(frames) < 3:
        return None

    try:

        # --------------------------------------------------------
        # Make sure dimensions match.
        # --------------------------------------------------------

        height, width = frames[0].shape[:2]

        normalized = []

        for frame in frames:

            if frame.shape[:2] != (height, width):

                frame = cv2.resize(
                    frame,
                    (width, height),
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
        #
        # A pulsing logo produces a localized region whose
        # brightness changes between frames.
        # --------------------------------------------------------

        temporal_std = np.std(
            stack,
            axis=0,
        )

        # --------------------------------------------------------
        # Frame-to-frame absolute difference.
        # --------------------------------------------------------

        diffs = np.abs(
            stack[1:] - stack[:-1]
        )

        temporal_diff = np.mean(
            diffs,
            axis=0,
        )

        # --------------------------------------------------------
        # Normalize both signals to 0-255.
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
        # Clean small noise.
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
        # Find connected components.
        # --------------------------------------------------------

        contours, _ = cv2.findContours(
            combined,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        image_area = width * height

        candidates = []

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 20:
                continue

            ratio = area / image_area

            # Reject huge regions. The watermark should not
            # consume a significant part of the sports graphic.
            if ratio > MAX_MASK_AREA_RATIO:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if w < 5 or h < 5:
                continue

            # Score using temporal intensity.
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
                + float(np.mean(roi_diff))
            )

            candidates.append(
                (
                    score,
                    contour,
                )
            )

        if not candidates:
            return None

        # --------------------------------------------------------
        # Take the strongest candidates.
        #
        # A pulsing logo may consist of several disconnected
        # pieces, e.g. letters + icon.
        # --------------------------------------------------------

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected = candidates[:8]

        mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        total_area = 0

        for _, contour in selected:

            area = cv2.contourArea(contour)

            if total_area + area > (
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
        # Close gaps between pieces of the logo.
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

        mask = _expand_mask(mask)

        # --------------------------------------------------------
        # Final sanity check.
        # --------------------------------------------------------

        final_ratio = (
            cv2.countNonZero(mask)
            / image_area
        )

        if final_ratio > MAX_MASK_AREA_RATIO:
            logger.warning(
                "⚠️ Learned mask is too large: %.2f%%",
                final_ratio * 100,
            )
            return None

        logger.info(
            "🧠 Learned fixed pulsing mask: %.2f%% of frame",
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

        result = cv2.inpaint(
            frame,
            mask,
            INPAINT_RADIUS,
            cv2.INPAINT_TELEA,
        )

        return result

    except Exception:
        logger.exception(
            "⚠️ Inpainting failed; returning original frame."
        )
        return frame


# ============================================================
# STATIC IMAGE PROCESSING
# ============================================================

def process_static_image(
    image_bytes: bytes,
):
    """
    Process a normal image.

    The original image is preserved except for a detected
    watermark region.
    """

    try:

        source = Image.open(
            io.BytesIO(image_bytes)
        )

        source.load()

        frame = _pil_to_bgr(source)

        # OCR can directly identify @cappersfree.
        ocr_mask = detect_cappersfree_text(
            frame
        )

        if ocr_mask is None:

            logger.info(
                "ℹ️ No @cappersfree text detected."
            )

            # Do not blindly modify a static image when we have
            # no reliable watermark detection.
            output_frame = frame

        else:

            logger.info(
                "🔎 @cappersfree watermark detected."
            )

            output_frame = remove_watermark(
                frame,
                ocr_mask,
            )

        output = _bgr_to_pil(
            output_frame
        )

        # Preserve PNG-like output.
        buffer = io.BytesIO()

        output.save(
            buffer,
            format="PNG",
        )

        return BufferedInputFile(
            buffer.getvalue(),
            filename="cleaned.png",
        )

    except Exception:
        logger.exception(
            "❌ STATIC IMAGE PROCESSING FAILED"
        )
        return None


# ============================================================
# GIF PROCESSING
# ============================================================

def process_gif(
    image_bytes: bytes,
):
    """
    Process an actual GIF.

    Every frame is inspected.

    The CF logo is expected to remain in one fixed position
    while pulsing/fading.

    A single learned mask is applied to every frame so that
    the animation remains synchronized.
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

        for frame in ImageSequence.Iterator(source):

            converted = frame.convert(
                "RGB"
            )

            frames.append(
                _pil_to_bgr(converted)
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
        # Learn fixed pulsing CF region.
        # --------------------------------------------------------

        learned_mask = learn_fixed_pulsing_mask(
            frames
        )

        # --------------------------------------------------------
        # OCR detection across several frames.
        #
        # This helps if the literal @cappersfree text is visible
        # during some frames but not others.
        # --------------------------------------------------------

        ocr_mask = None

        sample_indexes = np.linspace(
            0,
            len(frames) - 1,
            min(8, len(frames)),
            dtype=int,
        )

        for index in sample_indexes:

            candidate = detect_cappersfree_text(
                frames[index]
            )

            ocr_mask = combine_masks(
                ocr_mask,
                candidate,
            )

        final_mask = combine_masks(
            learned_mask,
            ocr_mask,
        )

        if final_mask is None:

            logger.info(
                "ℹ️ No sufficiently reliable CF watermark detected."
            )

            # Return original GIF rather than damaging it.
            return BufferedInputFile(
                image_bytes,
                filename="cleaned.gif",
            )

        # --------------------------------------------------------
        # Apply the same learned mask to every frame.
        # --------------------------------------------------------

        cleaned_frames = []

        for index, frame in enumerate(frames):

            cleaned = remove_watermark(
                frame,
                final_mask,
            )

            cleaned_frames.append(
                _bgr_to_pil(cleaned)
            )

            logger.debug(
                "🧹 Cleaned GIF frame %d/%d",
                index + 1,
                len(frames),
            )

        # --------------------------------------------------------
        # Rebuild GIF.
        # --------------------------------------------------------

        output = io.BytesIO()

        first = cleaned_frames[0]

        first.save(
            output,
            format="GIF",
            save_all=True,
            append_images=cleaned_frames[1:],
            duration=durations,
            loop=loop,
            optimize=False,
        )

        gif_bytes = output.getvalue()

        logger.info(
            "✅ Cleaned GIF generated: %d bytes",
            len(gif_bytes),
        )

        return BufferedInputFile(
            gif_bytes,
            filename="cleaned.gif",
        )

    except Exception:
        logger.exception(
            "❌ GIF PROCESSING FAILED"
        )
        return None


# ============================================================
# VIDEO / TELEGRAM GIF-AS-MP4
# ============================================================

def _extract_mp4_frames(
    video_bytes: bytes,
):
    """
    Extract frames from Telegram's MP4 representation of a GIF.

    Returns:

        frames
        durations
        width
        height
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
                tmp_path / "input.mp4"
            )

            output_pattern = str(
                tmp_path / "frame_%06d.png"
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
                            image.convert("RGB")
                        )
                    )

            # ----------------------------------------------------
            # Obtain FPS.
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


def process_mp4_gif(
    video_bytes: bytes,
):
    """
    Telegram often sends GIFs as MP4.

    Process every frame exactly like a GIF and rebuild
    the result as an animated GIF.
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
        "🎞️ MP4/GIF: processing %d frames.",
        len(frames),
    )

    learned_mask = learn_fixed_pulsing_mask(
        frames
    )

    ocr_mask = None

    sample_indexes = np.linspace(
        0,
        len(frames) - 1,
        min(8, len(frames)),
        dtype=int,
    )

    for index in sample_indexes:

        candidate = detect_cappersfree_text(
            frames[index]
        )

        ocr_mask = combine_masks(
            ocr_mask,
            candidate,
        )

    final_mask = combine_masks(
        learned_mask,
        ocr_mask,
    )

    # No reliable watermark = don't alter.
    if final_mask is None:

        logger.info(
            "ℹ️ No reliable CF watermark found in MP4/GIF."
        )

        # Since Telegram supplied MP4, return it unchanged.
        return BufferedInputFile(
            video_bytes,
            filename="cleaned.mp4",
        )

    cleaned_frames = []

    for index, frame in enumerate(frames):

        cleaned = remove_watermark(
            frame,
            final_mask,
        )

        cleaned_frames.append(
            _bgr_to_pil(cleaned)
        )

    output = io.BytesIO()

    cleaned_frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=cleaned_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )

    logger.info(
        "✅ MP4/GIF cleaned and rebuilt as GIF."
    )

    return BufferedInputFile(
        output.getvalue(),
        filename="cleaned.gif",
    )


# ============================================================
# MAIN MEDIA ENTRY POINT
# ============================================================

async def regenerate_image_from_bytes(
    image_bytes: bytes,
):
    """
    Compatibility function.

    IMPORTANT:

    This no longer regenerates anything.

    It only removes a detected CF/@cappersfree watermark.

        ORIGINAL
            ↓
        DETECTION
            ↓
        MASK
            ↓
        INPAINT
            ↓
        CLEANED ORIGINAL

    No OpenAI.
    No image generation.
    No reconstruction.
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

        if image_bytes.startswith(
            b"GIF87a"
        ) or image_bytes.startswith(
            b"GIF89a"
        ):

            logger.info(
                "🎞️ GIF detected."
            )

            return process_gif(
                image_bytes
            )

        # --------------------------------------------------------
        # Telegram GIF-as-MP4.
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
        # Normal image.
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
