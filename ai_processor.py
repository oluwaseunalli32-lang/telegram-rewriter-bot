import os
import re
import base64
import time
import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from io import BytesIO

from PIL import Image
from openai import OpenAI
from aiogram.types import BufferedInputFile


# ============================================================
# OPENAI
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is missing!")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# CAPTION CONFIGURATION
# ============================================================

# This comes from Render:
#
# NEW_MENTION=@PrimeAnalysiss
#
# You do NOT need to put @PrimeAnalysiss inside this file.

OLD_MENTION = os.getenv(
    "OLD_MENTION",
    "@cappersfree",
).strip()

NEW_MENTION = os.getenv(
    "NEW_MENTION",
    "",
).strip()


# Make sure the new username has @
if NEW_MENTION and not NEW_MENTION.startswith("@"):
    NEW_MENTION = "@" + NEW_MENTION


# ============================================================
# OPENAI COOLDOWNS
# ============================================================

_last_vision_call = 0.0
_vision_cooldown = 3

_last_generation_call = 0.0
_generation_cooldown = 3


# ============================================================
# STARTUP CONFIG LOG
# ============================================================

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("🤖 AI PROCESSOR STARTED")
print("👤 OLD_MENTION:", repr(OLD_MENTION))
print("👤 NEW_MENTION:", repr(NEW_MENTION))
print(
    "🧠 Caption AI rewriting: DISABLED"
)
print(
    "✏️ Caption processing: EXACT REPLACEMENT"
)
print(
    "🖼️ Image pipeline: VISION → DECONSTRUCTION → GENERATION"
)
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ============================================================
# CAPTION PROCESSING
# ============================================================

def replace_username(text: str) -> str:
    """
    Process the Telegram caption.

    ONLY does two things:

    1. Remove every literal '*'
    2. Replace @cappersfree with NEW_MENTION

    Nothing is paraphrased or rewritten by AI.
    """

    if not text:
        return text

    result = text

    # --------------------------------------------------------
    # Remove Markdown asterisks.
    #
    # Example:
    #
    # **SBK**
    #
    # becomes:
    #
    # SBK
    # --------------------------------------------------------

    result = result.replace("*", "")

    # --------------------------------------------------------
    # Replace old username.
    #
    # We do this AFTER removing '*', so these all work:
    #
    # @cappersfree
    # @**cappersfree
    # @***cappersfree
    # --------------------------------------------------------

    if NEW_MENTION:

        pattern = re.escape(
            OLD_MENTION
        )

        result = re.sub(
            pattern,
            NEW_MENTION,
            result,
            flags=re.IGNORECASE,
        )

    else:

        print(
            "⚠️ NEW_MENTION is empty. "
            "Username was not replaced."
        )

    return result


async def rewrite_text(
    original_text: str,
) -> str:
    """
    Kept with the same function name so main.py
    does not need to change.

    NO DeepSeek.
    NO OpenAI.
    NO paraphrasing.

    Only:
        remove *
        replace @cappersfree
    """

    result = replace_username(
        original_text
    )

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📝 CAPTION PROCESSING")
    print("📝 ORIGINAL:", repr(original_text))
    print("📝 FINAL:   ", repr(result))
    print("👤 OLD:     ", repr(OLD_MENTION))
    print("👤 NEW:     ", repr(NEW_MENTION))
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return result


# ============================================================
# IMAGE PREPARATION
# ============================================================

def resize_for_vision(
    img: Image.Image,
) -> Image.Image:
    """
    Resize image while preserving aspect ratio.
    """

    max_dimension = 2048

    if max(img.size) <= max_dimension:
        return img

    scale = (
        max_dimension
        / max(img.size)
    )

    new_size = (
        max(
            1,
            int(
                img.width * scale
            ),
        ),
        max(
            1,
            int(
                img.height * scale
            ),
        ),
    )

    return img.resize(
        new_size,
        Image.Resampling.LANCZOS,
    )


def _is_video_container(data: bytes) -> bool:
    """Detect MP4/MOV/WebM containers before Pillow sees them."""
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return True
    return data.startswith(b"\x1a\x45\xdf\xa3")


def _extract_video_contact_sheet(video_bytes: bytes):
    """
    Telegram GIFs are frequently MP4 files. Extract representative
    frames with the ffmpeg binary bundled by imageio-ffmpeg.
    """
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        logger.error(
            "❌ imageio-ffmpeg is missing. "
            "Add imageio-ffmpeg to requirements.txt."
        )
        return None

    try:
        ffmpeg = get_ffmpeg_exe()

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "telegram_media"
            output_pattern = str(Path(tmp) / "frame_%02d.jpg")

            input_path.write_bytes(video_bytes)

            command = [
                ffmpeg, "-y", "-i", str(input_path),
                "-vf",
                "fps=1/1,scale=1536:-2:force_original_aspect_ratio=decrease",
                "-frames:v", "6", output_pattern,
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=90,
            )

            if result.returncode != 0:
                logger.error(
                    "❌ FFmpeg failed: %s",
                    result.stderr.decode("utf-8", errors="replace")[-3000:],
                )
                return None

            frame_paths = sorted(Path(tmp).glob("frame_*.jpg"))

            if not frame_paths:
                logger.error("❌ FFmpeg produced no frames.")
                return None

            frames = []
            for frame_path in frame_paths:
                try:
                    with Image.open(frame_path) as frame:
                        frames.append(
                            resize_for_vision(
                                frame.convert("RGB")
                            ).copy()
                        )
                except Exception:
                    logger.exception(
                        "⚠️ Could not read extracted frame %s",
                        frame_path,
                    )

            if not frames:
                return None

            logger.info(
                "🎞️ Video/GIF detected: extracted %d representative frames.",
                len(frames),
            )

            columns = min(3, len(frames))
            rows = (len(frames) + columns - 1) // columns
            cell_width = max(frame.width for frame in frames)
            cell_height = max(frame.height for frame in frames)
            padding = 20

            sheet = Image.new(
                "RGB",
                (
                    columns * cell_width + (columns + 1) * padding,
                    rows * cell_height + (rows + 1) * padding,
                ),
                "white",
            )

            for index, frame in enumerate(frames):
                x = padding + (index % columns) * (cell_width + padding)
                y = padding + (index // columns) * (cell_height + padding)
                sheet.paste(frame, (x, y))

            return resize_for_vision(sheet)

    except Exception:
        logger.exception("❌ VIDEO/GIF FRAME EXTRACTION FAILED")
        return None


def prepare_image_for_vision(image_bytes: bytes):
    """
    Prepare Telegram photos, images, animated GIFs, and GIF-as-MP4
    for OpenAI Vision.
    """
    if not image_bytes:
        logger.error("❌ prepare_image_for_vision received empty bytes.")
        return None

    logger.info(
        "🔬 Media signature: %s",
        bytes(image_bytes[:16]).hex(" "),
    )

    try:
        # IMPORTANT: Telegram GIFs are often MP4/video documents.
        if _is_video_container(image_bytes):
            logger.info(
                "🎞️ Video container detected; extracting frames..."
            )
            return _extract_video_contact_sheet(image_bytes)

        source = Image.open(BytesIO(image_bytes))
        source.load()

        if not bool(getattr(source, "is_animated", False)):
            rgba = source.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(
                rgba,
                (0, 0),
                rgba.getchannel("A"),
            )
            return resize_for_vision(background)

        total_frames = int(getattr(source, "n_frames", 1))
        sample_count = min(6, total_frames)

        if sample_count <= 1:
            frame_indexes = [0]
        else:
            frame_indexes = [
                round(
                    i * (total_frames - 1) / (sample_count - 1)
                )
                for i in range(sample_count)
            ]

        frames = []

        for frame_number in frame_indexes:
            try:
                source.seek(frame_number)
                rgba = source.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(
                    rgba,
                    (0, 0),
                    rgba.getchannel("A"),
                )
                frames.append(
                    resize_for_vision(background).copy()
                )
            except Exception:
                logger.exception(
                    "⚠️ Could not read GIF frame %s",
                    frame_number,
                )

        if not frames:
            logger.error("❌ No GIF frames could be extracted.")
            return None

        logger.info(
            "🎞️ Animated GIF detected: %d total frames; using %d representative frames.",
            total_frames,
            len(frames),
        )

        columns = min(3, len(frames))
        rows = (len(frames) + columns - 1) // columns
        cell_width = max(frame.width for frame in frames)
        cell_height = max(frame.height for frame in frames)
        padding = 20

        sheet = Image.new(
            "RGB",
            (
                columns * cell_width + (columns + 1) * padding,
                rows * cell_height + (rows + 1) * padding,
            ),
            "white",
        )

        for index, frame in enumerate(frames):
            x = padding + (index % columns) * (cell_width + padding)
            y = padding + (index // columns) * (cell_height + padding)
            sheet.paste(frame, (x, y))

        return resize_for_vision(sheet)

    except Exception:
        logger.exception("❌ IMAGE PREPARATION FAILED")
        return None


# ============================================================
# OPENAI VISION
# ============================================================

async def describe_image_bytes(
    image_bytes: bytes,
):
    """
    Send the image/contact sheet to OpenAI Vision
    and receive a detailed reconstruction specification.
    """

    global _last_vision_call

    if not image_bytes:

        print(
            "❌ Vision received empty image bytes."
        )

        return None

    # ========================================================
    # COOLDOWN
    # ========================================================

    elapsed = (
        time.time()
        - _last_vision_call
    )

    if elapsed < _vision_cooldown:

        wait = (
            _vision_cooldown
            - elapsed
        )

        await asyncio.sleep(
            wait
        )

    _last_vision_call = time.time()

    # ========================================================
    # PREPARE IMAGE
    # ========================================================

    try:

        image = prepare_image_for_vision(
            image_bytes
        )

        if image is None:

            return None

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=92,
        )

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode(
            "utf-8"
        )

        image_url = (
            "data:image/jpeg;base64,"
            + encoded
        )

    except Exception as e:

        print(
            "❌ Failed to prepare Vision image:"
        )

        print(
            type(e).__name__,
            str(e),
        )

        return None

    # ========================================================
    # VISION PROMPT
    # ========================================================

    prompt = """
Analyze this image carefully and create a detailed
reconstruction specification for a NEW image.

The purpose is to understand the visual design and legitimate
content so another image-generation model can create a new
similar graphic.

IMPORTANT:

Do NOT reproduce:
- watermarks
- usernames
- @handles
- social-media handles
- channel branding
- third-party logos
- third-party brand marks

Describe those areas generically instead.

Do NOT include the actual watermark or username in your
reconstruction specification.

============================================================
1. CANVAS
============================================================

Describe:
- orientation
- approximate aspect ratio
- layout dimensions
- overall proportions

============================================================
2. MAIN CONTENT
============================================================

Describe all legitimate visible content:

- people
- athletes
- teams
- sports
- objects
- equipment
- backgrounds
- locations
- cards
- panels
- icons
- graphics

============================================================
3. LEGITIMATE TEXT
============================================================

Transcribe visible legitimate text accurately.

Include:
- headings
- matchups
- scores
- odds
- picks
- dates
- times
- labels
- numbers

Do NOT include:
- usernames
- @handles
- watermarks
- channel names
- branding

Never invent text that is not visible.

============================================================
4. COMPOSITION
============================================================

Describe:
- top section
- center section
- bottom section
- left/right placement
- alignment
- spacing
- margins
- cards
- panels
- borders
- dividers

============================================================
5. TYPOGRAPHY
============================================================

Describe:
- font style
- font weight
- approximate size hierarchy
- uppercase/lowercase
- alignment
- text color
- outlines
- shadows
- glow
- spacing

============================================================
6. COLORS
============================================================

Describe:
- background
- primary colors
- secondary colors
- accent colors
- gradients
- highlights
- shadows

============================================================
7. BACKGROUND
============================================================

Describe:
- texture
- lighting
- atmosphere
- depth
- blur
- patterns
- gradients
- shadows
- environmental details

============================================================
8. GRAPHIC ELEMENTS
============================================================

Describe legitimate:
- icons
- arrows
- shapes
- lines
- badges
- panels
- borders
- decorative elements

============================================================
9. ANIMATION
============================================================

If the image is a contact sheet from a GIF:

Describe:
- what changes between frames
- movement
- transitions
- which elements remain fixed
- which elements change
- overall animation concept

============================================================
10. FINAL GENERATION SPECIFICATION
============================================================

Finish with a detailed specification that another image
generator can use to create a NEW image with:

- similar composition
- similar hierarchy
- similar visual style
- similar color relationships
- similar legitimate information
- similar overall design language

while excluding watermarks, usernames, @handles,
channel branding and third-party logos.

Do not invent information.
"""

    # ========================================================
    # VISION REQUEST
    # ========================================================

    max_retries = 3

    for attempt in range(
        max_retries
    ):

        try:

            print(
                f"🔍 OpenAI Vision request "
                f"{attempt + 1}/{max_retries}"
            )

            response = (
                openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url,
                                        "detail": "high",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=4000,
                )
            )

            description = (
                response
                .choices[0]
                .message
                .content
            )

            if not description:

                print(
                    "❌ Vision returned empty text."
                )

                return None

            description = description.strip()

            print(
                f"✅ Vision deconstruction complete "
                f"({len(description)} chars)"
            )

            print(
                "🧩 Vision preview:"
            )

            print(
                description[:1200]
            )

            return description

        except Exception as e:

            status_code = getattr(
                e,
                "status_code",
                None,
            )

            print(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            print(
                "❌ OPENAI VISION ERROR"
            )

            print(
                "❌ Type:",
                type(e).__name__,
            )

            print(
                "❌ Error:",
                str(e),
            )

            print(
                "❌ Status:",
                status_code,
            )

            if hasattr(
                e,
                "body",
            ):

                print(
                    "❌ Body:",
                    getattr(
                        e,
                        "body",
                        None,
                    ),
                )

            print(
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            if (
                status_code == 429
                and attempt < max_retries - 1
            ):

                wait = (
                    2 ** attempt
                )

                print(
                    f"⏳ Retrying Vision in "
                    f"{wait}s..."
                )

                await asyncio.sleep(
                    wait
                )

                continue

            return None

    return None


# ============================================================
# IMAGE GENERATION
# ============================================================

async def generate_image_from_description(
    description: str,
):
    """
    Generate a NEW image from the Vision
    reconstruction specification.
    """

    global _last_generation_call

    if not description:

        print(
            "❌ Empty reconstruction description."
        )

        return None

    # ========================================================
    # COOLDOWN
    # ========================================================

    elapsed = (
        time.time()
        - _last_generation_call
    )

    if elapsed < _generation_cooldown:

        wait = (
            _generation_cooldown
            - elapsed
        )

        await asyncio.sleep(
            wait
        )

    _last_generation_call = time.time()

    # ========================================================
    # GENERATION PROMPT
    # ========================================================

    generation_prompt = f"""
Create a NEW professional sports graphic using the
reconstruction specification below.

This is a NEW generated image, not an edited copy.

============================================================
DESIGN
============================================================

Preserve the described:
- composition
- hierarchy
- spacing
- visual structure
- colors
- typography style
- background treatment
- legitimate information

============================================================
TEXT
============================================================

Make legitimate text as accurate and readable as possible.

Do not invent:
- scores
- odds
- dates
- teams
- players
- picks
- numbers

============================================================
EXCLUDE
============================================================

Do NOT include:
- watermarks
- usernames
- @handles
- social-media handles
- channel branding
- third-party logos
- third-party brand marks

Any area that previously contained such material should
become a clean part of the new design.

============================================================
QUALITY
============================================================

Create:
- sharp text
- professional typography
- clean spacing
- strong hierarchy
- polished sports-graphic appearance
- coherent colors
- clean background
- high visual quality

============================================================
RECONSTRUCTION SPECIFICATION
============================================================

{description}
"""

    # ========================================================
    # REQUEST
    # ========================================================

    try:

        print(
            "🎨 Sending reconstruction "
            "specification to image generation..."
        )

        print(
            f"📝 Prompt length: "
            f"{len(generation_prompt)} characters"
        )

        response = (
            openai_client.images.generate(
                model="gpt-image-2",
                prompt=generation_prompt,
                n=1,
            )
        )

        # ====================================================
        # CHECK RESPONSE
        # ====================================================

        if not response:

            print(
                "❌ OpenAI returned no response."
            )

            return None

        if not response.data:

            print(
                "❌ OpenAI returned no image data."
            )

            return None

        image_data = response.data[0]

        # ====================================================
        # BASE64
        # ====================================================

        b64_json = getattr(
            image_data,
            "b64_json",
            None,
        )

        if b64_json:

            image_bytes = base64.b64decode(
                b64_json
            )

            print(
                f"✅ NEW IMAGE GENERATED "
                f"({len(image_bytes)} bytes)"
            )

            return BufferedInputFile(
                file=image_bytes,
                filename="regenerated.png",
            )

        # ====================================================
        # URL
        # ====================================================

        image_url = getattr(
            image_data,
            "url",
            None,
        )

        if image_url:

            print(
                "✅ NEW IMAGE GENERATED "
                "as URL."
            )

            return image_url

        print(
            "❌ OpenAI response contained "
            "neither b64_json nor URL."
        )

        return None

    except Exception as e:

        # ====================================================
        # THIS IS IMPORTANT
        # ====================================================
        #
        # The previous version only showed:
        #
        # 400 Bad Request
        #
        # This version prints the actual OpenAI error body.
        # ====================================================

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        print(
            "❌ OPENAI IMAGE GENERATION ERROR"
        )

        print(
            "❌ Exception type:",
            type(e).__name__,
        )

        print(
            "❌ Exception:",
            str(e),
        )

        print(
            "❌ Status code:",
            getattr(
                e,
                "status_code",
                None,
            ),
        )

        print(
            "❌ Error code:",
            getattr(
                e,
                "code",
                None,
            ),
        )

        print(
            "❌ Parameter:",
            getattr(
                e,
                "param",
                None,
            ),
        )

        body = getattr(
            e,
            "body",
            None,
        )

        if body:

            print(
                "❌ API ERROR BODY:"
            )

            print(
                body
            )

        response_obj = getattr(
            e,
            "response",
            None,
        )

        if response_obj:

            try:

                print(
                    "❌ API RESPONSE:"
                )

                print(
                    response_obj
                )

            except Exception:
                pass

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        return None


# ============================================================
# COMPLETE REGENERATION PIPELINE
# ============================================================

async def regenerate_image_from_bytes(
    image_bytes: bytes,
):
    """
    Complete pipeline:

        Telegram image/GIF
                ↓
        Prepare image
                ↓
        OpenAI Vision
                ↓
        Deconstruction
                ↓
        OpenAI image generation
                ↓
        NEW IMAGE
    """

    if not image_bytes:

        print(
            "❌ regenerate_image_from_bytes "
            "received no bytes."
        )

        return None

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        "🖼️ IMAGE RECREATION PIPELINE START"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    # ========================================================
    # STEP 1 — VISION
    # ========================================================

    print(
        "1️⃣ STEP 1/2 — "
        "OpenAI Vision deconstruction..."
    )

    try:

        description = (
            await describe_image_bytes(
                image_bytes
            )
        )

    except Exception as e:

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        print(
            "❌ VISION PIPELINE CRASHED"
        )

        print(
            "❌ Type:",
            type(e).__name__,
        )

        print(
            "❌ Error:",
            str(e),
        )

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        return None

    if not description:

        print(
            "❌ STEP 1 FAILED."
        )

        print(
            "❌ No reconstruction specification "
            "was returned."
        )

        return None

    # ========================================================
    # STEP 2 — GENERATION
    # ========================================================

    print(
        "2️⃣ STEP 2/2 — "
        "OpenAI image generation..."
    )

    try:

        generated = (
            await generate_image_from_description(
                description
            )
        )

    except Exception as e:

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        print(
            "❌ GENERATION PIPELINE CRASHED"
        )

        print(
            "❌ Type:",
            type(e).__name__,
        )

        print(
            "❌ Error:",
            str(e),
        )

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        return None

    if not generated:

        print(
            "❌ STEP 2 FAILED."
        )

        print(
            "❌ No regenerated image was produced."
        )

        return None

    print(
        "✅ NEW IMAGE CREATED"
    )

    print(
        "✅ IMAGE RECREATION PIPELINE COMPLETE"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return generated
