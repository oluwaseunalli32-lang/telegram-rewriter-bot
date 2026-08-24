import os
import io
import cv2
import time
import logging
import tempfile
import subprocess

import numpy as np

from pathlib import Path
from PIL import Image, ImageSequence


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WATERMARK_DIR = (
    BASE_DIR
    / "assets"
    / "watermarks"
)


# ============================================================
# WATERMARK SETTINGS
# ============================================================

# Example project structure:
#
# assets/
#   watermarks/
#       cf_logo.png
#       cappersfree.png
#
# You can put additional watermark templates here too.
#
# IMPORTANT:
# Transparent PNG templates are strongly recommended.

IMAGE_MATCH_THRESHOLD = float(
    os.getenv(
        "IMAGE_MATCH_THRESHOLD",
        "0.78",
    )
)

TEXT_MATCH_THRESHOLD = float(
    os.getenv(
        "TEXT_MATCH_THRESHOLD",
        "0.84",
    )
)

INPAINT_RADIUS = int(
    os.getenv(
        "INPAINT_RADIUS",
        "3",
    )
)

MASK_DILATION = int(
    os.getenv(
        "MASK_DILATION",
        "3",
    )
)


# ============================================================
# STARTUP
# ============================================================

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("🧹 WATERMARK REMOVAL PROCESSOR STARTED")
print("📁 Watermark directory:", WATERMARK_DIR)
print("🧠 OpenAI Vision: DISABLED")
print("🎨 Image generation: DISABLED")
print("🧹 Watermark removal: OPENCV INPAINTING")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ============================================================
# BASIC MEDIA DETECTION
# ============================================================

def is_video_container(data: bytes) -> bool:
    """
    Detect MP4/MOV/WebM.
    """

    if not data:
        return False

    # MP4/MOV/M4V/etc.
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return True

    # WebM / Matroska
    if data.startswith(
        b"\x1a\x45\xdf\xa3"
    ):
        return True

    return False


def is_animated_image(data: bytes) -> bool:
    """
    Detect animated GIF/WebP.
    """

    try:
        with Image.open(
            io.BytesIO(data)
        ) as image:

            return bool(
                getattr(
                    image,
                    "is_animated",
                    False,
                )
            )

    except Exception:
        return False


# ============================================================
# WATERMARK TEMPLATE LOADING
# ============================================================

def get_watermark_files():
    """
    Find all watermark template images.

    Every image inside:

        assets/watermarks/

    is treated as a watermark template.

    This means you can add:
        cf_logo.png
        cappersfree.png
        cf_logo_small.png
        another_watermark.png

    without changing this Python file.
    """

    if not WATERMARK_DIR.exists():

        logger.warning(
            "⚠️ Watermark directory does not exist: %s",
            WATERMARK_DIR,
        )

        return []

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
    }

    files = []

    for path in sorted(
        WATERMARK_DIR.iterdir()
    ):

        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed_extensions:
            continue

        files.append(path)

    return files


def load_watermark_template(
    path: Path,
):
    """
    Load watermark template.

    Returns:

        template_bgr
        alpha_mask
        filename
    """

    try:

        image = Image.open(path).convert(
            "RGBA"
        )

        rgba = np.array(image)

        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        # ----------------------------------------------------
        # Crop completely transparent edges.
        # ----------------------------------------------------

        alpha_pixels = np.where(
            alpha > 5
        )

        if len(alpha_pixels[0]) > 0:

            y1 = int(alpha_pixels[0].min())
            y2 = int(alpha_pixels[0].max()) + 1

            x1 = int(alpha_pixels[1].min())
            x2 = int(alpha_pixels[1].max()) + 1

            rgb = rgb[
                y1:y2,
                x1:x2,
            ]

            alpha = alpha[
                y1:y2,
                x1:x2,
            ]

        # ----------------------------------------------------
        # For template matching, transparent pixels are
        # composited onto white.
        # ----------------------------------------------------

        alpha_float = (
            alpha.astype(np.float32)
            / 255.0
        )

        white = np.full_like(
            rgb,
            255,
        )

        composite = (
            rgb.astype(np.float32)
            * alpha_float[:, :, None]
            +
            white.astype(np.float32)
            * (1.0 - alpha_float[:, :, None])
        )

        composite = np.clip(
            composite,
            0,
            255,
        ).astype(np.uint8)

        template_bgr = cv2.cvtColor(
            composite,
            cv2.COLOR_RGB2BGR,
        )

        # ----------------------------------------------------
        # Alpha becomes the actual area to remove.
        #
        # If the source has no transparency, remove the
        # complete template rectangle.
        # ----------------------------------------------------

        if np.max(alpha) <= 5:

            mask = np.ones(
                alpha.shape,
                dtype=np.uint8,
            ) * 255

        else:

            mask = np.where(
                alpha > 10,
                255,
                0,
            ).astype(np.uint8)

        return (
            template_bgr,
            mask,
            path.name,
        )

    except Exception:

        logger.exception(
            "❌ Could not load watermark template: %s",
            path,
        )

        return None


# ============================================================
# TEMPLATE MATCHING
# ============================================================

def _intersection_over_union(
    a,
    b,
):
    """
    Calculate IoU for two boxes.

    Box format:

        x1, y1, x2, y2
    """

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (
        (ix2 - ix1)
        * (iy2 - iy1)
    )

    area_a = (
        (ax2 - ax1)
        * (ay2 - ay1)
    )

    area_b = (
        (bx2 - bx1)
        * (by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def find_template_matches(
    image_bgr,
    template_bgr,
    threshold,
):
    """
    Find multiple occurrences of a watermark.

    Uses several scales because the watermark may not always
    have exactly the same size.
    """

    image_gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    template_gray_original = cv2.cvtColor(
        template_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    image_height, image_width = (
        image_gray.shape[:2]
    )

    original_height, original_width = (
        template_gray_original.shape[:2]
    )

    if (
        original_width < 4
        or original_height < 4
    ):
        return []

    # --------------------------------------------------------
    # Multiple scales.
    # --------------------------------------------------------

    scales = [
        0.35,
        0.45,
        0.55,
        0.65,
        0.75,
        0.85,
        1.00,
        1.15,
        1.30,
        1.50,
        1.75,
        2.00,
    ]

    candidates = []

    for scale in scales:

        width = max(
            4,
            int(
                original_width
                * scale
            ),
        )

        height = max(
            4,
            int(
                original_height
                * scale
            ),
        )

        if width >= image_width:
            continue

        if height >= image_height:
            continue

        template_gray = cv2.resize(
            template_gray_original,
            (width, height),
            interpolation=(
                cv2.INTER_AREA
                if scale < 1
                else cv2.INTER_LINEAR
            ),
        )

        try:

            result = cv2.matchTemplate(
                image_gray,
                template_gray,
                cv2.TM_CCOEFF_NORMED,
            )

        except Exception:

            logger.exception(
                "❌ Template matching failed."
            )

            continue

        # ----------------------------------------------------
        # Find several peaks at this scale.
        # ----------------------------------------------------

        for _ in range(20):

            min_value, max_value, min_location, max_location = (
                cv2.minMaxLoc(result)
            )

            if max_value < threshold:
                break

            x = int(max_location[0])
            y = int(max_location[1])

            box = (
                x,
                y,
                x + width,
                y + height,
            )

            candidates.append(
                (
                    float(max_value),
                    box,
                )
            )

            # Suppress this area so we can find another one.
            suppression_radius_x = max(
                4,
                width // 2,
            )

            suppression_radius_y = max(
                4,
                height // 2,
            )

            sx1 = max(
                0,
                x - suppression_radius_x,
            )

            sy1 = max(
                0,
                y - suppression_radius_y,
            )

            sx2 = min(
                result.shape[1],
                x
                + width
                + suppression_radius_x,
            )

            sy2 = min(
                result.shape[0],
                y
                + height
                + suppression_radius_y,
            )

            result[
                sy1:sy2,
                sx1:sx2,
            ] = -1

    # --------------------------------------------------------
    # Global non-maximum suppression.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    selected = []

    for score, box in candidates:

        duplicate = False

        for _, existing_box in selected:

            if (
                _intersection_over_union(
                    box,
                    existing_box,
                )
                > 0.35
            ):

                duplicate = True
                break

        if duplicate:
            continue

        selected.append(
            (
                score,
                box,
            )
        )

        if len(selected) >= 30:
            break

    return selected


# ============================================================
# MASK CREATION
# ============================================================

def add_match_to_mask(
    full_mask,
    match_box,
    template_mask,
):
    """
    Put a detected watermark's mask onto the full image mask.
    """

    x1, y1, x2, y2 = match_box

    image_height, image_width = (
        full_mask.shape[:2]
    )

    # Clip to image.
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(
        image_width,
        x2,
    )

    y2 = min(
        image_height,
        y2,
    )

    if x2 <= x1 or y2 <= y1:
        return

    width = x2 - x1
    height = y2 - y1

    resized_mask = cv2.resize(
        template_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    full_mask[
        y1:y2,
        x1:x2,
    ] = np.maximum(
        full_mask[
            y1:y2,
            x1:x2,
        ],
        resized_mask,
    )


def remove_watermarks_from_frame(
    frame_bgr,
):
    """
    Detect all configured watermark templates and remove
    every detected occurrence from one frame.
    """

    template_files = (
        get_watermark_files()
    )

    if not template_files:

        logger.error(
            "❌ No watermark templates found."
        )

        logger.error(
            "❌ Add watermark images to: %s",
            WATERMARK_DIR,
        )

        return None

    full_mask = np.zeros(
        frame_bgr.shape[:2],
        dtype=np.uint8,
    )

    total_matches = 0

    for template_path in template_files:

        loaded = load_watermark_template(
            template_path
        )

        if not loaded:
            continue

        (
            template_bgr,
            template_mask,
            template_name,
        ) = loaded

        template_height, template_width = (
            template_bgr.shape[:2]
        )

        # ----------------------------------------------------
        # Use a stronger threshold for tiny text templates.
        # ----------------------------------------------------

        if (
            template_width
            * template_height
            < 10000
        ):

            threshold = (
                TEXT_MATCH_THRESHOLD
            )

        else:

            threshold = (
                IMAGE_MATCH_THRESHOLD
            )

        matches = find_template_matches(
            frame_bgr,
            template_bgr,
            threshold,
        )

        if matches:

            logger.info(
                "🧹 %s: found %d occurrence(s)",
                template_name,
                len(matches),
            )

        for score, box in matches:

            logger.debug(
                "🧹 Match %s score=%.3f box=%s",
                template_name,
                score,
                box,
            )

            add_match_to_mask(
                full_mask,
                box,
                template_mask,
            )

            total_matches += 1

    if total_matches == 0:

        # No watermark detected.
        # IMPORTANT:
        # Return the original frame unchanged.
        return frame_bgr

    # --------------------------------------------------------
    # Slightly expand the mask.
    #
    # This helps remove edges around the logo/text.
    # --------------------------------------------------------

    if MASK_DILATION > 0:

        kernel_size = (
            MASK_DILATION * 2
        ) + 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size,
            ),
        )

        full_mask = cv2.dilate(
            full_mask,
            kernel,
            iterations=1,
        )

    # --------------------------------------------------------
    # OpenCV inpainting.
    #
    # This does NOT generate a new image.
    # It reconstructs the masked pixels from surrounding
    # pixels.
    # --------------------------------------------------------

    cleaned = cv2.inpaint(
        frame_bgr,
        full_mask,
        INPAINT_RADIUS,
        cv2.INPAINT_NS,
    )

    logger.info(
        "🧹 Removed %d watermark occurrence(s).",
        total_matches,
    )

    return cleaned


# ============================================================
# STILL IMAGE
# ============================================================

def process_still_image(
    image_bytes: bytes,
):
    """
    Process a normal image.
    """

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGBA")

        rgba = np.array(image)

        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]

        # OpenCV works with BGR.
        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        cleaned_bgr = (
            remove_watermarks_from_frame(
                bgr
            )
        )

        if cleaned_bgr is None:
            return None

        cleaned_rgb = cv2.cvtColor(
            cleaned_bgr,
            cv2.COLOR_BGR2RGB,
        )

        cleaned_rgba = np.dstack(
            (
                cleaned_rgb,
                alpha,
            )
        )

        output_image = Image.fromarray(
            cleaned_rgba,
            "RGBA",
        )

        output = io.BytesIO()

        output_image.save(
            output,
            format="PNG",
            optimize=True,
        )

        return (
            output.getvalue(),
            "photo",
        )

    except Exception:

        logger.exception(
            "❌ STILL IMAGE PROCESSING FAILED"
        )

        return None


# ============================================================
# FFMPEG
# ============================================================

def get_ffmpeg():
    """
    Get the ffmpeg binary supplied by imageio-ffmpeg.
    """

    try:

        from imageio_ffmpeg import (
            get_ffmpeg_exe,
        )

        return get_ffmpeg_exe()

    except ImportError:

        logger.error(
            "❌ imageio-ffmpeg is missing."
        )

        return None


def encode_frames_to_mp4(
    frames_dir: Path,
    fps: float,
    output_path: Path,
):
    """
    Encode processed PNG frames into MP4.
    """

    ffmpeg = get_ffmpeg()

    if not ffmpeg:
        return False

    command = [
        ffmpeg,
        "-y",

        "-framerate",
        str(max(1.0, fps)),

        "-i",
        str(
            frames_dir
            / "frame_%06d.png"
        ),

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "18",

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
        timeout=300,
    )

    if result.returncode != 0:

        logger.error(
            "❌ FFmpeg encoding failed:"
        )

        logger.error(
            result.stderr.decode(
                "utf-8",
                errors="replace",
            )[-4000:]
        )

        return False

    return (
        output_path.exists()
        and output_path.stat().st_size > 0
    )


# ============================================================
# ANIMATED GIF
# ============================================================

def process_gif(
    image_bytes: bytes,
):
    """
    Process every GIF frame.

    The result is MP4 rather than GIF so it can be included
    together with photos in a Telegram media group.
    """

    try:

        source = Image.open(
            io.BytesIO(image_bytes)
        )

        frames = []

        durations = []

        for frame in ImageSequence.Iterator(
            source
        ):

            rgba = frame.convert(
                "RGBA"
            )

            rgba_array = np.array(
                rgba
            )

            rgb = rgba_array[:, :, :3]

            alpha = rgba_array[:, :, 3]

            bgr = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            )

            cleaned_bgr = (
                remove_watermarks_from_frame(
                    bgr
                )
            )

            if cleaned_bgr is None:
                return None

            cleaned_rgb = cv2.cvtColor(
                cleaned_bgr,
                cv2.COLOR_BGR2RGB,
            )

            cleaned_rgba = np.dstack(
                (
                    cleaned_rgb,
                    alpha,
                )
            )

            cleaned = Image.fromarray(
                cleaned_rgba,
                "RGBA",
            )

            frames.append(
                cleaned
            )

            duration = frame.info.get(
                "duration",
                source.info.get(
                    "duration",
                    100,
                ),
            )

            durations.append(
                max(
                    10,
                    int(duration or 100),
                )
            )

        if not frames:
            return None

        # Average GIF frame duration.
        average_duration = (
            sum(durations)
            / len(durations)
        )

        fps = (
            1000.0
            / average_duration
        )

        with tempfile.TemporaryDirectory() as tmp:

            tmp_dir = Path(tmp)

            for index, frame in enumerate(
                frames
            ):

                frame_path = (
                    tmp_dir
                    / f"frame_{index:06d}.png"
                )

                frame.save(
                    frame_path,
                    format="PNG",
                )

            output_path = (
                tmp_dir
                / "cleaned.mp4"
            )

            success = (
                encode_frames_to_mp4(
                    tmp_dir,
                    fps,
                    output_path,
                )
            )

            if not success:
                return None

            return (
                output_path.read_bytes(),
                "video",
            )

    except Exception:

        logger.exception(
            "❌ GIF PROCESSING FAILED"
        )

        return None


# ============================================================
# VIDEO / MP4
# ============================================================

def process_video(
    video_bytes: bytes,
):
    """
    Process every video frame and produce a cleaned MP4.
    """

    try:

        with tempfile.TemporaryDirectory() as tmp:

            tmp_dir = Path(tmp)

            input_path = (
                tmp_dir
                / "input.mp4"
            )

            input_path.write_bytes(
                video_bytes
            )

            capture = cv2.VideoCapture(
                str(input_path)
            )

            if not capture.isOpened():

                logger.error(
                    "❌ Could not open video."
                )

                return None

            fps = capture.get(
                cv2.CAP_PROP_FPS
            )

            if not fps or fps <= 0:
                fps = 15.0

            frame_number = 0

            while True:

                success, frame = (
                    capture.read()
                )

                if not success:
                    break

                cleaned = (
                    remove_watermarks_from_frame(
                        frame
                    )
                )

                if cleaned is None:

                    capture.release()

                    return None

                frame_path = (
                    tmp_dir
                    / (
                        f"frame_"
                        f"{frame_number:06d}.png"
                    )
                )

                cv2.imwrite(
                    str(frame_path),
                    cleaned,
                )

                frame_number += 1

            capture.release()

            if frame_number == 0:

                logger.error(
                    "❌ Video contained no frames."
                )

                return None

            output_path = (
                tmp_dir
                / "cleaned.mp4"
            )

            success = (
                encode_frames_to_mp4(
                    tmp_dir,
                    fps,
                    output_path,
                )
            )

            if not success:
                return None

            return (
                output_path.read_bytes(),
                "video",
            )

    except Exception:

        logger.exception(
            "❌ VIDEO PROCESSING FAILED"
        )

        return None


# ============================================================
# PUBLIC FUNCTION
# ============================================================

async def remove_watermarks_from_bytes(
    media_bytes: bytes,
):
    """
    Main public function.

    Returns:

        (
            cleaned_bytes,
            "photo"
        )

    or:

        (
            cleaned_bytes,
            "video"
        )

    or None on failure.

    NO OpenAI.
    NO image generation.
    NO caption processing.
    """

    if not media_bytes:

        logger.error(
            "❌ Empty media received."
        )

        return None

    logger.info(
        "🧹 WATERMARK REMOVAL START"
    )

    logger.info(
        "📦 Original bytes: %d",
        len(media_bytes),
    )

    # --------------------------------------------------------
    # Video / MP4 / WebM
    # --------------------------------------------------------

    if is_video_container(
        media_bytes
    ):

        logger.info(
            "🎞️ Video container detected."
        )

        return await asyncio.to_thread(
            process_video,
            media_bytes,
        )

    # --------------------------------------------------------
    # Animated GIF / WebP
    # --------------------------------------------------------

    if is_animated_image(
        media_bytes
    ):

        logger.info(
            "🎞️ Animated image detected."
        )

        return await asyncio.to_thread(
            process_gif,
            media_bytes,
        )

    # --------------------------------------------------------
    # Normal image
    # --------------------------------------------------------

    logger.info(
        "🖼️ Static image detected."
    )

    return await asyncio.to_thread(
        process_still_image,
        media_bytes,
    )
