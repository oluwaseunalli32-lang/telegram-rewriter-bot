import os
import re
import cv2
import time
import logging
import tempfile
import subprocess

from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image

logger = logging.getLogger("ai_processor")


# ============================================================
# CONFIGURATION
# ============================================================

OLD_MENTION = os.getenv(
    "OLD_MENTION",
    "@cappersfree",
).strip()

# Folder containing your CF logo template.
#
# Example:
#
# project/
#   main.py
#   ai_processor.py
#   cf_templates/
#       cf_logo.png
#
TEMPLATE_DIR = Path(
    os.getenv(
        "CF_TEMPLATE_DIR",
        "cf_templates",
    )
)

LOGO_TEMPLATE_PATH = TEMPLATE_DIR / "cf_logo.png"


# ============================================================
# WATERMARK SETTINGS
# ============================================================

# Padding around detected @cappersfree text.
TEXT_PADDING = 10

# Padding around detected logo.
LOGO_PADDING = 8

# OCR is intentionally disabled unless pytesseract is installed.
#
# This keeps the script functional even when OCR is unavailable.
try:
    import pytesseract

    OCR_AVAILABLE = True

except ImportError:
    pytesseract = None
    OCR_AVAILABLE = False


# ============================================================
# STARTUP
# ============================================================

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("🧹 WATERMARK REMOVAL PROCESSOR STARTED")
print("🧹 OpenAI image generation: DISABLED")
print("🧹 OpenAI Vision: DISABLED")
print("🧹 Image recreation: DISABLED")
print("🧹 Caption modification: DISABLED")
print("🧹 Watermark removal: ENABLED")
print("👤 Watermark:", repr(OLD_MENTION))
print("🔎 OCR available:", OCR_AVAILABLE)
print("🖼️ Logo template:", str(LOGO_TEMPLATE_PATH))
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ============================================================
# CAPTION
# ============================================================

async def rewrite_text(original_text: str) -> str:
    """
    IMPORTANT:

    Caption is returned EXACTLY as received.

    No:
        - rewriting
        - username replacement
        - markdown removal
        - AI processing
        - formatting changes
    """

    return original_text or ""


# ============================================================
# BASIC IMAGE DETECTION
# ============================================================

def is_video_container(data: bytes) -> bool:
    """
    Detect MP4/MOV/WebM.
    """

    if len(data) >= 12 and data[4:8] == b"ftyp":
        return True

    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return True

    return False


def load_pil_image(data: bytes):
    """
    Load a normal image from bytes.
    """

    try:

        image = Image.open(
            BytesIO(data)
        )

        image.load()

        return image

    except Exception:

        logger.exception(
            "❌ Could not open image."
        )

        return None


# ============================================================
# OCR
# ============================================================

def normalize_ocr_text(text: str) -> str:
    """
    Normalize OCR output so variations such as:

        @cappersfree
        cappersfree
        @ cappersfree
        cappers free

    can be detected.
    """

    if not text:
        return ""

    text = text.lower()

    text = text.replace(
        "@",
        "",
    )

    text = re.sub(
        r"[^a-z0-9]",
        "",
        text,
    )

    return text


def find_cappersfree_text_boxes(
    image: np.ndarray,
):
    """
    Find @cappersfree using OCR.

    Returns rectangles:

        [(x, y, w, h), ...]
    """

    if not OCR_AVAILABLE:
        return []

    boxes = []

    try:

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        data = pytesseract.image_to_data(
            rgb,
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )

        count = len(
            data.get("text", [])
        )

        for i in range(count):

            raw_text = (
                data["text"][i]
                or ""
            ).strip()

            normalized = normalize_ocr_text(
                raw_text
            )

            if "cappersfree" not in normalized:
                continue

            try:

                confidence = float(
                    data["conf"][i]
                )

            except Exception:

                confidence = 0

            # Ignore extremely uncertain OCR.
            if confidence < 20:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            if w <= 0 or h <= 0:
                continue

            boxes.append(
                (
                    x,
                    y,
                    w,
                    h,
                )
            )

    except Exception:

        logger.exception(
            "⚠️ OCR failed."
        )

    return boxes


# ============================================================
# LOGO TEMPLATE
# ============================================================

def load_logo_template():
    """
    Load cf_templates/cf_logo.png.

    If it doesn't exist, logo matching is simply skipped.
    """

    if not LOGO_TEMPLATE_PATH.exists():

        logger.warning(
            "⚠️ CF logo template not found: %s",
            LOGO_TEMPLATE_PATH,
        )

        return None

    try:

        template = cv2.imread(
            str(LOGO_TEMPLATE_PATH),
            cv2.IMREAD_COLOR,
        )

        if template is None:

            logger.warning(
                "⚠️ Could not read CF logo template."
            )

            return None

        return template

    except Exception:

        logger.exception(
            "❌ Failed loading CF logo template."
        )

        return None


LOGO_TEMPLATE = load_logo_template()


def add_padding_to_box(
    box,
    image_width,
    image_height,
    padding,
):
    """
    Expand rectangle safely.
    """

    x, y, w, h = box

    x1 = max(
        0,
        x - padding,
    )

    y1 = max(
        0,
        y - padding,
    )

    x2 = min(
        image_width,
        x + w + padding,
    )

    y2 = min(
        image_height,
        y + h + padding,
    )

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )


def find_logo_boxes(
    image: np.ndarray,
):
    """
    Find the CF logo using multi-scale template matching.

    The logo is optional. If it is not present, nothing is removed.
    """

    if LOGO_TEMPLATE is None:
        return []

    boxes = []

    try:

        image_gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        template_gray = cv2.cvtColor(
            LOGO_TEMPLATE,
            cv2.COLOR_BGR2GRAY,
        )

        original_h, original_w = (
            template_gray.shape[:2]
        )

        if original_w < 10 or original_h < 10:
            return []

        image_h, image_w = image_gray.shape[:2]

        # Search several reasonable scales.
        scales = [
            0.35,
            0.45,
            0.55,
            0.65,
            0.75,
            0.85,
            1.0,
            1.15,
            1.30,
            1.50,
        ]

        candidates = []

        for scale in scales:

            width = int(
                original_w * scale
            )

            height = int(
                original_h * scale
            )

            if width < 12 or height < 12:
                continue

            if width >= image_w:
                continue

            if height >= image_h:
                continue

            resized_template = cv2.resize(
                template_gray,
                (
                    width,
                    height,
                ),
                interpolation=cv2.INTER_AREA,
            )

            result = cv2.matchTemplate(
                image_gray,
                resized_template,
                cv2.TM_CCOEFF_NORMED,
            )

            threshold = 0.72

            locations = np.where(
                result >= threshold
            )

            for y, x in zip(
                locations[0],
                locations[1],
            ):

                score = float(
                    result[y, x]
                )

                candidates.append(
                    (
                        score,
                        x,
                        y,
                        width,
                        height,
                    )
                )

        # Highest-confidence matches first.
        candidates.sort(
            reverse=True,
            key=lambda item: item[0],
        )

        for (
            score,
            x,
            y,
            w,
            h,
        ) in candidates:

            # Avoid selecting the same logo repeatedly.
            new_box = (
                x,
                y,
                w,
                h,
            )

            overlaps = False

            for existing in boxes:

                if boxes_overlap(
                    new_box,
                    existing,
                ):

                    overlaps = True
                    break

            if overlaps:
                continue

            boxes.append(
                new_box
            )

            # Usually there will be one CF graphic.
            # Allow up to three in case a post contains several.
            if len(boxes) >= 3:
                break

            logger.info(
                "🟢 CF logo detected "
                "(confidence %.2f)",
                score,
            )

    except Exception:

        logger.exception(
            "⚠️ Logo detection failed."
        )

    return boxes


def boxes_overlap(
    a,
    b,
):
    """
    Determine whether two rectangles overlap.
    """

    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2 = ax + aw
    ay2 = ay + ah

    bx2 = bx + bw
    by2 = by + bh

    return not (
        ax2 <= bx
        or bx2 <= ax
        or ay2 <= by
        or by2 <= ay
    )


# ============================================================
# WATERMARK MASK
# ============================================================

def build_watermark_mask(
    image: np.ndarray,
):
    """
    Create a mask containing only:

        @cappersfree
        CF logo

    Everything else stays untouched.
    """

    height, width = image.shape[:2]

    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    text_boxes = find_cappersfree_text_boxes(
        image
    )

    for box in text_boxes:

        padded = add_padding_to_box(
            box,
            width,
            height,
            TEXT_PADDING,
        )

        x, y, w, h = padded

        cv2.rectangle(
            mask,
            (
                x,
                y,
            ),
            (
                x + w,
                y + h,
            ),
            255,
            -1,
        )

        logger.info(
            "🔤 @cappersfree watermark detected "
            "at x=%d y=%d w=%d h=%d",
            x,
            y,
            w,
            h,
        )

    # --------------------------------------------------------
    # LOGO
    # --------------------------------------------------------

    logo_boxes = find_logo_boxes(
        image
    )

    for box in logo_boxes:

        padded = add_padding_to_box(
            box,
            width,
            height,
            LOGO_PADDING,
        )

        x, y, w, h = padded

        cv2.rectangle(
            mask,
            (
                x,
                y,
            ),
            (
                x + w,
                y + h,
            ),
            255,
            -1,
        )

    return mask


# ============================================================
# INPAINT
# ============================================================

def remove_watermark_from_frame(
    frame: np.ndarray,
):
    """
    Remove only detected watermark regions.

    If nothing is detected, the original frame is returned
    unchanged.
    """

    mask = build_watermark_mask(
        frame
    )

    pixels = int(
        cv2.countNonZero(mask)
    )

    if pixels == 0:

        return frame

    logger.info(
        "🧹 Removing watermark from %d pixels.",
        pixels,
    )

    # Dilate slightly so antialiased edges are also removed.
    kernel = np.ones(
        (
            3,
            3,
        ),
        np.uint8,
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )

    cleaned = cv2.inpaint(
        frame,
        mask,
        5,
        cv2.INPAINT_TELEA,
    )

    return cleaned


# ============================================================
# STATIC IMAGE
# ============================================================

def clean_image_bytes(
    image_bytes: bytes,
):
    """
    Clean one normal image.

    Output:
        PNG bytes
    """

    image = load_pil_image(
        image_bytes
    )

    if image is None:
        return None

    rgba = image.convert(
        "RGBA"
    )

    # Preserve transparency correctly.
    if "A" in rgba.getbands():

        background = Image.new(
            "RGBA",
            rgba.size,
            (
                255,
                255,
                255,
                0,
            ),
        )

        background.alpha_composite(
            rgba
        )

        rgba = background

    rgb = Image.new(
        "RGB",
        rgba.size,
        "white",
    )

    rgb.paste(
        rgba,
        mask=rgba.getchannel("A"),
    )

    frame = cv2.cvtColor(
        np.array(rgb),
        cv2.COLOR_RGB2BGR,
    )

    cleaned = remove_watermark_from_frame(
        frame
    )

    cleaned_rgb = cv2.cvtColor(
        cleaned,
        cv2.COLOR_BGR2RGB,
    )

    output = BytesIO()

    Image.fromarray(
        cleaned_rgb
    ).save(
        output,
        format="PNG",
    )

    return output.getvalue()


# ============================================================
# VIDEO / GIF
# ============================================================

def clean_video_bytes(
    video_bytes: bytes,
):
    """
    Process every video/GIF frame.

    Output:
        MP4 bytes

    The video itself is NOT regenerated.
    Frames are simply decoded, watermark-cleaned,
    and encoded again.
    """

    with tempfile.TemporaryDirectory() as tmp:

        tmp_path = Path(tmp)

        input_path = (
            tmp_path / "input_media"
        )

        output_path = (
            tmp_path / "cleaned.mp4"
        )

        input_path.write_bytes(
            video_bytes
        )

        ffmpeg = None

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

        # ----------------------------------------------------
        # Decode video to PNG frames.
        # ----------------------------------------------------

        frames_dir = (
            tmp_path / "frames"
        )

        frames_dir.mkdir()

        decode_command = [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-vsync",
            "0",
            str(
                frames_dir / "frame_%08d.png"
            ),
        ]

        result = subprocess.run(
            decode_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )

        if result.returncode != 0:

            logger.error(
                "❌ FFmpeg decode failed: %s",
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
                "❌ No video frames found."
            )

            return None

        # ----------------------------------------------------
        # Process every frame.
        # ----------------------------------------------------

        for frame_path in frame_paths:

            frame = cv2.imread(
                str(frame_path),
                cv2.IMREAD_COLOR,
            )

            if frame is None:
                continue

            cleaned = (
                remove_watermark_from_frame(
                    frame
                )
            )

            cv2.imwrite(
                str(frame_path),
                cleaned,
            )

        # ----------------------------------------------------
        # Rebuild MP4.
        # ----------------------------------------------------
        #
        # Using the source video's FPS.
        # FFmpeg's default framerate for image sequences
        # would otherwise be wrong.
        #
        # We first read FPS from ffprobe.
        # ----------------------------------------------------

        fps = get_video_fps(
            ffmpeg,
            input_path,
        )

        if fps <= 0:
            fps = 30.0

        encode_command = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(
                frames_dir / "frame_%08d.png"
            ),
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
                "❌ FFmpeg encode failed: %s",
                result.stderr.decode(
                    "utf-8",
                    errors="replace",
                )[-3000:],
            )

            return None

        if not output_path.exists():

            return None

        return output_path.read_bytes()


def get_video_fps(
    ffmpeg,
    input_path,
):
    """
    Obtain source FPS using ffmpeg.
    """

    try:

        command = [
            ffmpeg,
            "-i",
            str(input_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )

        text = result.stderr.decode(
            "utf-8",
            errors="replace",
        )

        match = re.search(
            r"(\d+(?:\.\d+)?)\s*fps",
            text,
        )

        if match:

            return float(
                match.group(1)
            )

    except Exception:

        logger.exception(
            "⚠️ Could not determine video FPS."
        )

    return 30.0


# ============================================================
# PUBLIC FUNCTION
# ============================================================

async def remove_watermark_from_bytes(
    media_bytes: bytes,
):
    """
    Main public function.

    Automatically handles:

        JPG
        JPEG
        PNG
        WEBP
        GIF
        MP4
        MOV
        WEBM

    No image generation.
    No Vision.
    No caption processing.
    """

    if not media_bytes:

        logger.error(
            "❌ Empty media bytes."
        )

        return None

    if is_video_container(
        media_bytes
    ):

        logger.info(
            "🎞️ Video/GIF detected."
        )

        return await asyncio.to_thread(
            clean_video_bytes,
            media_bytes,
        )

    logger.info(
        "🖼️ Static image detected."
    )

    return await asyncio.to_thread(
        clean_image_bytes,
        media_bytes,
    )
