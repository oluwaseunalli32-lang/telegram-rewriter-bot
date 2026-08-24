import os
import re
import io
import cv2
import asyncio
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
# WATERMARK TEMPLATE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WATERMARK_DIR = (
    BASE_DIR
    / "assets"
    / "watermarks"
)

CF_LOGO_PATH = (
    WATERMARK_DIR
    / "cf_logo.png"
)


# ============================================================
# RED / FADED-RED SETTINGS
# ============================================================

# Strong red.
STRONG_RED_S_MIN = 55
STRONG_RED_V_MIN = 45

# Faded red.
FADED_RED_S_MIN = 25
FADED_RED_V_MIN = 28

# Red channel must dominate the other channels.
RED_DOMINANCE_MIN = 12

# Minimum connected component.
MIN_RED_COMPONENT_AREA = 5

# Ignore extremely large red regions.
MAX_RED_COMPONENT_RATIO = 0.08

# Join individual red letters into text-like regions.
RED_TEXT_KERNEL_WIDTH = 31
RED_TEXT_KERNEL_HEIGHT = 7

# Final mask expansion.
MASK_DILATION = 5

# OpenCV inpainting radius.
INPAINT_RADIUS = 5


# ============================================================
# CF LOGO SETTINGS
# ============================================================

LOGO_MATCH_THRESHOLD = 0.70

LOGO_SCALES = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
    1.15,
    1.30,
    1.50,
    1.75,
    2.00,
]

MAX_LOGO_MATCHES = 5

# Safety limit for total pixels being altered.
MAX_TOTAL_MASK_RATIO = 0.15


# ============================================================
# GIF PULSING SETTINGS
# ============================================================

TEMPORAL_DIFF_THRESHOLD = 16

TEMPORAL_MIN_AREA = 25

TEMPORAL_MAX_AREA_RATIO = 0.08

TEMPORAL_CLOSE_WIDTH = 15

TEMPORAL_CLOSE_HEIGHT = 15


# ============================================================
# STARTUP
# ============================================================

logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
logger.info("🧹 WATERMARK PROCESSOR STARTED")
logger.info("🧹 OpenAI Vision: DISABLED")
logger.info("🧹 Image generation: DISABLED")
logger.info("🔴 Strong/faded-red detection: ENABLED")
logger.info("🟢 CF logo template detection: ENABLED")
logger.info("🎞️ GIF pulsing detection: ENABLED")
logger.info("🧹 Direct OpenCV inpainting: ENABLED")
logger.info("📁 CF logo path: %s", CF_LOGO_PATH)
logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ============================================================
# CAPTION PROCESSING
# ============================================================

def replace_username(text: str) -> str:
    """
    Existing caption behavior.

    1. Remove *
    2. Replace OLD_MENTION with NEW_MENTION

    Nothing else.
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
        "👤 OLD:      %r",
        OLD_MENTION,
    )

    logger.info(
        "👤 NEW:      %r",
        NEW_MENTION,
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return result


# ============================================================
# MEDIA HELPERS
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


def pil_to_bgr(
    image: Image.Image,
) -> np.ndarray:

    rgb = image.convert(
        "RGB"
    )

    return cv2.cvtColor(
        np.asarray(rgb),
        cv2.COLOR_RGB2BGR,
    )


def bgr_to_pil(
    frame: np.ndarray,
) -> Image.Image:

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(
        rgb
    )


# ============================================================
# RED / FADED-RED DETECTION
# ============================================================

def detect_red_pixels(
    frame: np.ndarray,
) -> np.ndarray:
    """
    Detect normal and faded red.

    Uses both HSV and actual RGB channel dominance, because a
    translucent red watermark may have relatively low saturation.
    """

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    b = frame[:, :, 0].astype(
        np.int16
    )

    g = frame[:, :, 1].astype(
        np.int16
    )

    r = frame[:, :, 2].astype(
        np.int16
    )

    red_dominance = (
        r
        - np.maximum(
            g,
            b,
        )
    )

    # --------------------------------------------------------
    # Strong red.
    # --------------------------------------------------------

    strong_1 = cv2.inRange(
        hsv,
        np.array(
            [
                0,
                STRONG_RED_S_MIN,
                STRONG_RED_V_MIN,
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                15,
                255,
                255,
            ],
            dtype=np.uint8,
        ),
    )

    strong_2 = cv2.inRange(
        hsv,
        np.array(
            [
                165,
                STRONG_RED_S_MIN,
                STRONG_RED_V_MIN,
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                179,
                255,
                255,
            ],
            dtype=np.uint8,
        ),
    )

    strong = cv2.bitwise_or(
        strong_1,
        strong_2,
    )

    # --------------------------------------------------------
    # Faded red.
    # --------------------------------------------------------

    faded_1 = cv2.inRange(
        hsv,
        np.array(
            [
                0,
                FADED_RED_S_MIN,
                FADED_RED_V_MIN,
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                18,
                255,
                255,
            ],
            dtype=np.uint8,
        ),
    )

    faded_2 = cv2.inRange(
        hsv,
        np.array(
            [
                162,
                FADED_RED_S_MIN,
                FADED_RED_V_MIN,
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                179,
                255,
                255,
            ],
            dtype=np.uint8,
        ),
    )

    faded = cv2.bitwise_or(
        faded_1,
        faded_2,
    )

    # --------------------------------------------------------
    # Red-channel dominance.
    # --------------------------------------------------------

    dominance = np.where(
        red_dominance >= RED_DOMINANCE_MIN,
        255,
        0,
    ).astype(
        np.uint8
    )

    combined = cv2.bitwise_or(
        strong,
        faded,
    )

    combined = cv2.bitwise_or(
        combined,
        dominance,
    )

    # Very low saturation pixels are generally grey/white and
    # should not be treated as red watermark pixels.
    combined[
        hsv[:, :, 1] < 18
    ] = 0

    return combined


def detect_red_text_mask(
    frame: np.ndarray,
) -> np.ndarray:

    red = detect_red_pixels(
        frame
    )

    h, w = red.shape

    text_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                RED_TEXT_KERNEL_WIDTH,
                RED_TEXT_KERNEL_HEIGHT,
            ),
        )
    )

    grouped = cv2.morphologyEx(
        red,
        cv2.MORPH_CLOSE,
        text_kernel,
    )

    grouped = cv2.dilate(
        grouped,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
        iterations=1,
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            grouped,
            connectivity=8,
        )
    )

    mask = np.zeros_like(
        grouped
    )

    for i in range(
        1,
        count,
    ):

        x = stats[
            i,
            cv2.CC_STAT_LEFT,
        ]

        y = stats[
            i,
            cv2.CC_STAT_TOP,
        ]

        ww = stats[
            i,
            cv2.CC_STAT_WIDTH,
        ]

        hh = stats[
            i,
            cv2.CC_STAT_HEIGHT,
        ]

        area = stats[
            i,
            cv2.CC_STAT_AREA,
        ]

        if area < MIN_RED_COMPONENT_AREA:
            continue

        if (
            area
            > h
            * w
            * MAX_RED_COMPONENT_RATIO
        ):
            continue

        # Watermark text should generally be wider than tall.
        if ww < 20:
            continue

        if hh > h * 0.25:
            continue

        aspect = (
            ww
            / max(
                1,
                hh,
            )
        )

        if aspect < 2.0:
            continue

        if ww > w * 0.70:
            continue

        mask[
            labels == i
        ] = 255

    return mask


# ============================================================
# CF LOGO TEMPLATE
# ============================================================

def load_cf_logo():

    if not CF_LOGO_PATH.exists():

        logger.warning(
            "⚠️ CF logo template not found: %s",
            CF_LOGO_PATH,
        )

        return None

    try:

        logo = Image.open(
            CF_LOGO_PATH
        ).convert(
            "RGBA"
        )

        rgba = np.asarray(
            logo
        )

        rgb = rgba[:, :, :3]

        alpha = rgba[:, :, 3]

        # ----------------------------------------------------
        # Crop transparent borders.
        # ----------------------------------------------------

        ys, xs = np.where(
            alpha > 10
        )

        if len(xs) > 0:

            rgb = rgb[
                ys.min():ys.max() + 1,
                xs.min():xs.max() + 1,
            ]

            alpha = alpha[
                ys.min():ys.max() + 1,
                xs.min():xs.max() + 1,
            ]

        alpha_f = (
            alpha.astype(
                np.float32
            )
            / 255.0
        )

        white = np.full_like(
            rgb,
            255,
        )

        composite = (
            rgb.astype(
                np.float32
            )
            * alpha_f[:, :, None]
            +
            white.astype(
                np.float32
            )
            * (
                1.0
                - alpha_f[:, :, None]
            )
        )

        composite = np.clip(
            composite,
            0,
            255,
        ).astype(
            np.uint8
        )

        bgr = cv2.cvtColor(
            composite,
            cv2.COLOR_RGB2BGR,
        )

        logo_mask = np.where(
            alpha > 10,
            255,
            0,
        ).astype(
            np.uint8
        )

        if cv2.countNonZero(
            logo_mask
        ) == 0:

            logo_mask = np.ones(
                alpha.shape,
                dtype=np.uint8,
            ) * 255

        logger.info(
            "✅ CF logo loaded: %sx%s",
            bgr.shape[1],
            bgr.shape[0],
        )

        return (
            bgr,
            logo_mask,
        )

    except Exception:

        logger.exception(
            "❌ Could not load CF logo."
        )

        return None


CF_LOGO = load_cf_logo()


# ============================================================
# LOGO TEMPLATE MATCHING
# ============================================================

def find_cf_logo_matches(
    frame: np.ndarray,
):

    if CF_LOGO is None:
        return []

    template, template_mask = (
        CF_LOGO
    )

    frame_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    template_gray = cv2.cvtColor(
        template,
        cv2.COLOR_BGR2GRAY,
    )

    frame_edges = cv2.Canny(
        frame_gray,
        50,
        150,
    )

    template_edges = cv2.Canny(
        template_gray,
        50,
        150,
    )

    image_h, image_w = (
        frame_gray.shape
    )

    template_h, template_w = (
        template_gray.shape
    )

    if (
        template_w < 8
        or template_h < 8
    ):
        return []

    candidates = []

    for scale in LOGO_SCALES:

        w = int(
            template_w
            * scale
        )

        h = int(
            template_h
            * scale
        )

        if w < 8 or h < 8:
            continue

        if w >= image_w or h >= image_h:
            continue

        resized = cv2.resize(
            template_edges,
            (
                w,
                h,
            ),
            interpolation=(
                cv2.INTER_AREA
                if scale < 1
                else cv2.INTER_LINEAR
            ),
        )

        try:

            result = cv2.matchTemplate(
                frame_edges,
                resized,
                cv2.TM_CCOEFF_NORMED,
            )

        except Exception:
            continue

        for _ in range(
            10
        ):

            _, max_value, _, max_location = (
                cv2.minMaxLoc(
                    result
                )
            )

            if (
                max_value
                < LOGO_MATCH_THRESHOLD
            ):
                break

            x, y = max_location

            candidates.append(
                (
                    float(max_value),
                    (
                        x,
                        y,
                        x + w,
                        y + h,
                    ),
                )
            )

            sx1 = max(
                0,
                x - w // 2,
            )

            sy1 = max(
                0,
                y - h // 2,
            )

            sx2 = min(
                result.shape[1],
                x + w + w // 2,
            )

            sy2 = min(
                result.shape[0],
                y + h + h // 2,
            )

            result[
                sy1:sy2,
                sx1:sx2,
            ] = -1

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    selected = []

    for score, box in candidates:

        x1, y1, x2, y2 = box

        duplicate = False

        for _, existing in selected:

            ex1, ey1, ex2, ey2 = (
                existing
            )

            ix1 = max(
                x1,
                ex1,
            )

            iy1 = max(
                y1,
                ey1,
            )

            ix2 = min(
                x2,
                ex2,
            )

            iy2 = min(
                y2,
                ey2,
            )

            if (
                ix2 > ix1
                and iy2 > iy1
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

        if len(selected) >= MAX_LOGO_MATCHES:
            break

    if selected:

        logger.info(
            "🟢 CF logo matches found: %d",
            len(selected),
        )

        for score, _ in selected:

            logger.info(
                "🟢 CF logo confidence: %.3f",
                score,
            )

    return selected


def add_logo_matches_to_mask(
    frame: np.ndarray,
    mask: np.ndarray,
):

    if CF_LOGO is None:
        return

    _, template_mask = (
        CF_LOGO
    )

    matches = find_cf_logo_matches(
        frame
    )

    for _, box in matches:

        x1, y1, x2, y2 = box

        x1 = max(
            0,
            x1,
        )

        y1 = max(
            0,
            y1,
        )

        x2 = min(
            frame.shape[1],
            x2,
        )

        y2 = min(
            frame.shape[0],
            y2,
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):
            continue

        width = x2 - x1
        height = y2 - y1

        resized_mask = cv2.resize(
            template_mask,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        resized_mask = cv2.dilate(
            resized_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (5, 5),
            ),
            iterations=1,
        )

        mask[
            y1:y2,
            x1:x2,
        ] = np.maximum(
            mask[
                y1:y2,
                x1:x2,
            ],
            resized_mask,
        )


# ============================================================
# GIF PULSE DETECTION
# ============================================================

def learn_pulsing_cf_region(
    frames,
):

    if len(frames) < 3:
        return None

    h, w = frames[0].shape[:2]

    gray_frames = []

    for frame in frames:

        if frame.shape[:2] != (
            h,
            w,
        ):

            frame = cv2.resize(
                frame,
                (
                    w,
                    h,
                ),
            )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (
                5,
                5,
            ),
            0,
        )

        gray_frames.append(
            gray
        )

    difference = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.float32,
    )

    for i in range(
        1,
        len(gray_frames),
    ):

        difference += (
            cv2.absdiff(
                gray_frames[i],
                gray_frames[i - 1],
            )
            .astype(
                np.float32
            )
        )

    difference /= (
        len(gray_frames) - 1
    )

    dynamic = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.uint8,
    )

    dynamic[
        difference
        >= TEMPORAL_DIFF_THRESHOLD
    ] = 255

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            TEMPORAL_CLOSE_WIDTH,
            TEMPORAL_CLOSE_HEIGHT,
        ),
    )

    dynamic = cv2.morphologyEx(
        dynamic,
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    dynamic = cv2.morphologyEx(
        dynamic,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                5,
                5,
            ),
        ),
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            dynamic,
            connectivity=8,
        )
    )

    mask = np.zeros_like(
        dynamic
    )

    for i in range(
        1,
        count,
    ):

        x = stats[
            i,
            cv2.CC_STAT_LEFT,
        ]

        y = stats[
            i,
            cv2.CC_STAT_TOP,
        ]

        ww = stats[
            i,
            cv2.CC_STAT_WIDTH,
        ]

        hh = stats[
            i,
            cv2.CC_STAT_HEIGHT,
        ]

        area = stats[
            i,
            cv2.CC_STAT_AREA,
        ]

        if area < TEMPORAL_MIN_AREA:
            continue

        if (
            area
            >
            h
            * w
            * TEMPORAL_MAX_AREA_RATIO
        ):
            continue

        if ww > w * 0.35:
            continue

        if hh > h * 0.35:
            continue

        cv2.rectangle(
            mask,
            (
                x,
                y,
            ),
            (
                x + ww,
                y + hh,
            ),
            255,
            -1,
        )

    if cv2.countNonZero(
        mask
    ) == 0:

        return None

    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                7,
                7,
            ),
        ),
        iterations=1,
    )

    ratio = (
        cv2.countNonZero(mask)
        / (
            h * w
        )
    )

    if ratio > MAX_TOTAL_MASK_RATIO:

        logger.warning(
            "⚠️ Learned GIF mask too large: %.2f%%",
            ratio * 100,
        )

        return None

    logger.info(
        "🧠 Learned pulsing CF region: %.2f%% of frame",
        ratio * 100,
    )

    return mask


# ============================================================
# MASK VALIDATION
# ============================================================

def validate_mask(
    mask: np.ndarray | None,
    frame_shape,
):

    if mask is None:
        return None

    total_pixels = (
        frame_shape[0]
        * frame_shape[1]
    )

    mask_pixels = cv2.countNonZero(
        mask
    )

    if mask_pixels == 0:
        return None

    ratio = (
        mask_pixels
        / total_pixels
    )

    if ratio > MAX_TOTAL_MASK_RATIO:

        logger.warning(
            "⚠️ Final mask too large: %.2f%%",
            ratio * 100,
        )

        return None

    return mask


# ============================================================
# INPAINT
# ============================================================

def inpaint_frame(
    frame: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:

    mask = validate_mask(
        mask,
        frame.shape,
    )

    if mask is None:
        return frame

    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                MASK_DILATION,
                MASK_DILATION,
            ),
        ),
        iterations=1,
    )

    return cv2.inpaint(
        frame,
        mask,
        INPAINT_RADIUS,
        cv2.INPAINT_TELEA,
    )


# ============================================================
# BUILD MASK FOR ONE STATIC FRAME
# ============================================================

def build_frame_mask(
    frame: np.ndarray,
):

    mask = np.zeros(
        frame.shape[:2],
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # RED / FADED RED
    # --------------------------------------------------------

    red_mask = detect_red_text_mask(
        frame
    )

    mask = cv2.bitwise_or(
        mask,
        red_mask,
    )

    red_pixels = cv2.countNonZero(
        red_mask
    )

    if red_pixels:

        logger.info(
            "🔴 Red/faded-red watermark candidates: %d pixels",
            red_pixels,
        )

    # --------------------------------------------------------
    # CF LOGO
    # --------------------------------------------------------

    before = cv2.countNonZero(
        mask
    )

    add_logo_matches_to_mask(
        frame,
        mask,
    )

    after = cv2.countNonZero(
        mask
    )

    if after > before:

        logger.info(
            "🟢 CF logo mask added: %d pixels",
            after - before,
        )

    return validate_mask(
        mask,
        frame.shape,
    )


# ============================================================
# STATIC IMAGE
# ============================================================

def process_static_image(
    image_bytes: bytes,
):

    try:

        source = Image.open(
            io.BytesIO(image_bytes)
        )

        source.load()

        frame = pil_to_bgr(
            source
        )

        mask = build_frame_mask(
            frame
        )

        if mask is None:

            logger.info(
                "ℹ️ No reliable watermark detected. "
                "Returning original image unchanged."
            )

            return (
                image_bytes,
                "photo",
            )

        cleaned = inpaint_frame(
            frame,
            mask,
        )

        output_image = bgr_to_pil(
            cleaned
        )

        output = io.BytesIO()

        original_format = (
            source.format
            or "PNG"
        ).upper()

        if original_format in {
            "JPG",
            "JPEG",
        }:

            output_image.save(
                output,
                format="JPEG",
                quality=97,
            )

        elif original_format == "WEBP":

            output_image.save(
                output,
                format="WEBP",
                quality=97,
            )

        else:

            output_image.save(
                output,
                format="PNG",
            )

        cleaned_bytes = (
            output.getvalue()
        )

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

            frames.append(
                pil_to_bgr(
                    frame.convert(
                        "RGB"
                    )
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
            return None

        logger.info(
            "🎞️ GIF contains %d frame(s).",
            len(frames),
        )

        # --------------------------------------------------------
        # Learn pulsing CF region.
        # --------------------------------------------------------

        temporal_mask = (
            learn_pulsing_cf_region(
                frames
            )
        )

        if temporal_mask is not None:

            logger.info(
                "🧠 Pulsing CF mask learned."
            )

        cleaned_frames = []

        for index, frame in enumerate(
            frames
        ):

            mask = build_frame_mask(
                frame
            )

            if temporal_mask is not None:

                if mask is None:

                    mask = (
                        temporal_mask.copy()
                    )

                else:

                    mask = cv2.bitwise_or(
                        mask,
                        temporal_mask,
                    )

            cleaned = inpaint_frame(
                frame,
                mask,
            )

            cleaned_frames.append(
                bgr_to_pil(
                    cleaned
                )
            )

            logger.info(
                "🎞️ Frame %d/%d processed.",
                index + 1,
                len(frames),
            )

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
            "✅ GIF cleaned: %d bytes",
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
# MP4 / TELEGRAM GIF
# ============================================================

def process_mp4(
    video_bytes: bytes,
):

    try:

        from imageio_ffmpeg import (
            get_ffmpeg_exe,
        )

        ffmpeg = get_ffmpeg_exe()

    except Exception:

        logger.exception(
            "❌ imageio-ffmpeg unavailable."
        )

        return None

    try:

        with tempfile.TemporaryDirectory() as tmp:

            tmp_path = Path(tmp)

            input_path = (
                tmp_path
                / "input.mp4"
            )

            input_path.write_bytes(
                video_bytes
            )

            frames_dir = (
                tmp_path
                / "frames"
            )

            frames_dir.mkdir()

            frame_pattern = (
                frames_dir
                / "frame_%06d.png"
            )

            extract_command = [
                ffmpeg,
                "-y",
                "-i",
                str(input_path),
                "-vsync",
                "0",
                str(frame_pattern),
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
                return None

            frames = []

            for path in frame_paths:

                frame = cv2.imread(
                    str(path),
                    cv2.IMREAD_COLOR,
                )

                if frame is not None:

                    frames.append(
                        frame
                    )

            if not frames:
                return None

            logger.info(
                "🎞️ MP4 contains %d frame(s).",
                len(frames),
            )

            temporal_mask = (
                learn_pulsing_cf_region(
                    frames
                )
            )

            processed_pattern = (
                frames_dir
                / "processed_%06d.png"
            )

            for index, frame in enumerate(
                frames,
                start=1,
            ):

                mask = build_frame_mask(
                    frame
                )

                if temporal_mask is not None:

                    if mask is None:

                        mask = (
                            temporal_mask.copy()
                        )

                    else:

                        mask = cv2.bitwise_or(
                            mask,
                            temporal_mask,
                        )

                cleaned = inpaint_frame(
                    frame,
                    mask,
                )

                path = (
                    frames_dir
                    / (
                        f"processed_{index:06d}.png"
                    )
                )

                cv2.imwrite(
                    str(path),
                    cleaned,
                )

            output_path = (
                tmp_path
                / "cleaned.mp4"
            )

            encode_command = [
                ffmpeg,
                "-y",
                "-framerate",
                "30",
                "-i",
                str(processed_pattern),
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
                encode_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )

            if result.returncode != 0:

                logger.error(
                    "❌ FFmpeg rebuild failed: %s",
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
                "✅ MP4 cleaned: %d bytes",
                len(cleaned_bytes),
            )

            return (
                cleaned_bytes,
                "video",
            )

    except Exception:

        logger.exception(
            "❌ MP4 PROCESSING FAILED"
        )

        return None


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

async def remove_watermarks_from_bytes(
    image_bytes: bytes,
    filename: str = "",
    mime_type: str = "",
):
    """
    Main entry point used by main.py.

    IMPORTANT:
    This ALWAYS returns an aiogram BufferedInputFile.

    It does NOT return:
        (bytes, "photo")

    This fixes the aiogram validation error from the previous
    deployment.
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

    try:

        # ====================================================
        # ACTUAL GIF
        # ====================================================

        if is_gif(
            image_bytes
        ):

            logger.info(
                "🎞️ Actual GIF detected."
            )

            result = await asyncio.to_thread(
                process_gif,
                image_bytes,
            )

            if not result:
                return None

            cleaned_bytes, _ = result

            return BufferedInputFile(
                cleaned_bytes,
                filename="watermark_removed.gif",
            )

        # ====================================================
        # MP4 / TELEGRAM GIF
        # ====================================================

        if is_video_container(
            image_bytes
        ):

            logger.info(
                "🎞️ MP4/video container detected."
            )

            result = await asyncio.to_thread(
                process_mp4,
                image_bytes,
            )

            if not result:
                return None

            cleaned_bytes, _ = result

            return BufferedInputFile(
                cleaned_bytes,
                filename="watermark_removed.mp4",
            )

        # ====================================================
        # STATIC IMAGE
        # ====================================================

        logger.info(
            "🖼️ Static image detected."
        )

        result = await asyncio.to_thread(
            process_static_image,
            image_bytes,
        )

        if not result:
            return None

        cleaned_bytes, _ = result

        return BufferedInputFile(
            cleaned_bytes,
            filename="watermark_removed.png",
        )

    except Exception:

        logger.exception(
            "❌ WATERMARK REMOVAL FAILED"
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

    return await remove_watermarks_from_bytes(
        image_bytes
    )
