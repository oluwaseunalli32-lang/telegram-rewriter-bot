import os
import re
import base64
import time
import asyncio
from io import BytesIO

from PIL import Image, ImageSequence
from openai import OpenAI
from aiogram.types import BufferedInputFile


# ============================================================
# API CLIENT
# ============================================================

# We deliberately do NOT use DeepSeek for username replacement.
#
# Username replacement must be deterministic:
#
#     @cappersfree
#          ↓
#     @YOUR_USERNAME
#
# An LLM should not be trusted to preserve odds, numbers,
# punctuation, team names, etc.

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ============================================================
# CONFIG
# ============================================================

NEW_MENTION = os.getenv(
    "NEW_MENTION",
    "",
).strip()

OLD_MENTION = os.getenv(
    "OLD_MENTION",
    "@cappersfree",
).strip()


# ============================================================
# RATE LIMIT / COOLDOWN
# ============================================================

_last_vision_call = 0.0
_vision_cooldown = 8

_last_generation_call = 0.0
_generation_cooldown = 8


# ============================================================
# TEXT PROCESSING
# ============================================================

def replace_username(text: str) -> str:
    """
    Replace ONLY the configured old username.

    Nothing else is changed.

    Example:

        @cappersfree
              ↓
        @myusername

    The function does NOT:
    - paraphrase
    - rewrite
    - summarize
    - change odds
    - change numbers
    - change team names
    - change punctuation
    - change other mentions
    """

    if not text:
        return text

    if not NEW_MENTION:
        return text

    if not OLD_MENTION:
        return text

    return re.sub(
        re.escape(OLD_MENTION),
        NEW_MENTION,
        text,
        flags=re.IGNORECASE,
    )


async def rewrite_text(original_text: str) -> str:
    """
    Kept with the old function name so main.py can simply import it.

    IMPORTANT:

    There is NO AI rewriting here.

    This is an exact deterministic username replacement.
    """

    return replace_username(
        original_text
    )


# ============================================================
# GIF / IMAGE PREPARATION
# ============================================================

def create_vision_image(
    image_bytes: bytes,
):
    """
    Prepare an image for OpenAI Vision.

    For normal images:
        return the original image converted to JPEG.

    For animated GIFs:
        sample multiple frames and create a contact sheet.

    This means Vision gets information about the animation
    rather than silently seeing only frame 1.
    """

    try:

        img = Image.open(
            BytesIO(image_bytes)
        )

        is_animated = getattr(
            img,
            "is_animated",
            False,
        )

        # --------------------------------------------------------
        # NORMAL STATIC IMAGE
        # --------------------------------------------------------

        if not is_animated:

            if img.mode not in (
                "RGB",
                "RGBA",
            ):
                img = img.convert("RGBA")

            if img.mode == "RGBA":

                background = Image.new(
                    "RGB",
                    img.size,
                    "white",
                )

                background.paste(
                    img,
                    mask=img.getchannel("A"),
                )

                img = background

            else:

                img = img.convert("RGB")

            return resize_for_vision(
                img
            )

        # --------------------------------------------------------
        # ANIMATED GIF
        # --------------------------------------------------------

        frames = []

        total_frames = getattr(
            img,
            "n_frames",
            1,
        )

        # We don't need every frame.
        # We want representative frames.
        sample_count = min(
            6,
            total_frames,
        )

        if sample_count <= 1:
            indexes = [0]
        else:
            indexes = [
                round(
                    i * (total_frames - 1)
                    / (sample_count - 1)
                )
                for i in range(sample_count)
            ]

        for index in indexes:

            try:

                img.seek(index)

                frame = img.convert(
                    "RGBA"
                )

                background = Image.new(
                    "RGB",
                    frame.size,
                    "white",
                )

                background.paste(
                    frame,
                    mask=frame.getchannel("A"),
                )

                frame = background

                frame.thumbnail(
                    (600, 600),
                    Image.Resampling.LANCZOS,
                )

                frames.append(
                    frame.copy()
                )

            except Exception as e:

                print(
                    f"⚠️ Could not read GIF frame "
                    f"{index}: {e}"
                )

        if not frames:
            return None

        # --------------------------------------------------------
        # CREATE CONTACT SHEET
        # --------------------------------------------------------

        columns = 3
        rows = (
            len(frames) + columns - 1
        ) // columns

        cell_width = max(
            frame.width
            for frame in frames
        )

        cell_height = max(
            frame.height
            for frame in frames
        )

        padding = 20

        sheet_width = (
            columns * cell_width
            + (columns + 1) * padding
        )

        sheet_height = (
            rows * cell_height
            + (rows + 1) * padding
        )

        sheet = Image.new(
            "RGB",
            (
                sheet_width,
                sheet_height,
            ),
            "white",
        )

        for i, frame in enumerate(frames):

            x = (
                padding
                + (i % columns)
                * (
                    cell_width
                    + padding
                )
            )

            y = (
                padding
                + (i // columns)
                * (
                    cell_height
                    + padding
                )
            )

            sheet.paste(
                frame,
                (x, y),
            )

        print(
            f"🎞️ Animated GIF detected: "
            f"{total_frames} frames"
        )

        print(
            f"🎞️ Sampled {len(frames)} frames "
            f"for Vision analysis"
        )

        return resize_for_vision(
            sheet
        )

    except Exception as e:

        print(
            f"❌ Image preparation error: {e}"
        )

        return None


def resize_for_vision(
    img: Image.Image,
):
    """
    Resize an image/contact sheet to a reasonable
    maximum size while preserving detail.
    """

    max_dimension = 2048

    if max(img.size) > max_dimension:

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

        img = img.resize(
            new_size,
            Image.Resampling.LANCZOS,
        )

    return img


# ============================================================
# OPENAI VISION / DECONSTRUCTION
# ============================================================

async def describe_image_bytes(
    image_bytes: bytes,
):
    """
    Send the original image/GIF to OpenAI Vision.

    For GIFs, multiple representative frames are supplied
    as a contact sheet.

    Vision produces a detailed reconstruction specification.

    Pipeline:

        ORIGINAL
            ↓
        VISION
            ↓
        RECONSTRUCTION SPEC
    """

    global _last_vision_call

    # --------------------------------------------------------
    # Cooldown
    # --------------------------------------------------------

    now = time.time()

    elapsed = (
        now - _last_vision_call
    )

    if elapsed < _vision_cooldown:

        wait = (
            _vision_cooldown
            - elapsed
        )

        print(
            f"⏳ Waiting {wait:.1f}s "
            f"before next Vision call..."
        )

        await asyncio.sleep(
            wait
        )

    _last_vision_call = time.time()

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    img = create_vision_image(
        image_bytes
    )

    if img is None:

        print(
            "❌ Could not prepare image "
            "for Vision."
        )

        return None

    buffered = BytesIO()

    img.save(
        buffered,
        format="JPEG",
        quality=95,
    )

    img_base64 = base64.b64encode(
        buffered.getvalue()
    ).decode(
        "utf-8"
    )

    data_url = (
        "data:image/jpeg;base64,"
        + img_base64
    )

    # ========================================================
    # VISION PROMPT
    # ========================================================

    vision_prompt = """
You are the visual analyst in an image recreation pipeline.

Your task is to carefully DECONSTRUCT the supplied image so
another image-generation model can create a NEW image with
the same general visual design and information structure.

The supplied image may be:
- a normal image
- an animated GIF represented by several sampled frames

If several frames are shown, analyze them collectively and
identify the visual elements that remain consistent as well
as any meaningful changes between frames.

IMPORTANT:

The final generated image must be a NEW independently
generated image.

Do not reproduce watermarks, usernames, social-media handles,
channel branding, logos, or other third-party branding.

Treat those elements as excluded visual elements.

Do NOT include the actual watermark, username, handle, logo,
or channel name in the reconstruction specification.

Instead, describe that area generically, for example:
"clean background area where branding appeared."

============================================================
ANALYZE THE IMAGE
============================================================

1. CANVAS
- orientation
- aspect ratio
- approximate dimensions
- overall proportions

2. MAIN SUBJECT / CONTENT
Describe all legitimate visible content:
- people
- players
- teams
- objects
- sports
- equipment
- locations
- scenes
- graphics
- cards
- panels
- icons
- symbols

3. TEXT CONTENT
Record legitimate visible text as accurately as possible.

Include:
- headings
- labels
- matchup information
- scores
- odds
- dates
- times
- picks
- numbers
- meaningful captions

Do NOT include usernames, social-media handles,
watermarks, channel names, logos, or branding.

Do not invent information that is not visible.

4. COMPOSITION
Describe:
- exact placement of major elements
- top / middle / bottom structure
- left / center / right alignment
- margins
- spacing
- hierarchy
- cards
- panels
- dividers
- borders
- frames
- arrows
- icons
- decorative elements

5. TYPOGRAPHY
Describe:
- font style
- approximate font family
- font weight
- uppercase/lowercase
- relative font sizes
- alignment
- spacing
- text colors
- outlines
- shadows
- glow
- other effects

6. COLOR PALETTE
Describe:
- dominant background colors
- secondary colors
- accent colors
- text colors
- gradients
- highlights
- shadows

7. BACKGROUND
Describe:
- texture
- grain
- lighting
- stadium/sports atmosphere
- patterns
- depth
- shadows
- glow
- blur
- environmental details

8. GRAPHIC ELEMENTS
Describe legitimate visual elements such as:
- players
- silhouettes
- sports equipment
- stadium elements
- abstract shapes
- arrows
- badges
- icons
- lines
- borders
- effects

9. GIF / ANIMATION ANALYSIS
If multiple frames are provided:
- describe what changes between frames
- describe movement
- describe transitions
- identify which elements remain fixed
- identify which elements animate
- describe the overall animation concept

10. FINAL RECREATION SPECIFICATION
Finish with a clear, detailed instruction set for an
image-generation model.

The goal is NOT to copy the original image pixel-for-pixel.

The goal is to create a new image that captures:
- the same type of composition
- similar visual hierarchy
- similar design language
- similar color relationships
- similar legitimate information
- similar overall visual impression

while excluding third-party watermarks,
usernames, handles, logos, and channel branding.

Do not shorten the analysis.
Do not invent information.
Do not add information that is not present.
"""

    # ========================================================
    # RETRIES
    # ========================================================

    max_retries = 4
    base_delay = 2

    for attempt in range(
        max_retries
    ):

        try:

            print(
                "🔍 Sending image to OpenAI Vision "
                f"(attempt {attempt + 1}/{max_retries})..."
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
                                    "text": vision_prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": data_url,
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
                .strip()
            )

            if not description:

                print(
                    "❌ OpenAI Vision returned "
                    "an empty description."
                )

                return None

            print(
                "✅ OpenAI Vision deconstruction "
                f"complete: {len(description)} characters"
            )

            print(
                "🧩 Deconstruction preview:"
            )

            print(
                description[:1000]
            )

            return description

        except Exception as e:

            status_code = getattr(
                e,
                "status_code",
                None,
            )

            if status_code == 429:

                wait = (
                    base_delay
                    * (
                        2 ** attempt
                    )
                )

                print(
                    "⚠️ OpenAI Vision rate limit "
                    f"(429). Retry in {wait}s"
                )

                await asyncio.sleep(
                    wait
                )

                continue

            print(
                f"❌ OpenAI Vision error: {e}"
            )

            return None

    print(
        "❌ OpenAI Vision failed "
        "after all retries."
    )

    return None


# ============================================================
# IMAGE GENERATION
# ============================================================

async def generate_image_from_description(
    description: str,
):
    """
    Generate a completely new image from the
    OpenAI Vision reconstruction specification.
    """

    global _last_generation_call

    if not description:

        print(
            "❌ Generation skipped: "
            "empty reconstruction specification."
        )

        return None

    if len(description.strip()) < 10:

        print(
            "❌ Generation skipped: "
            "reconstruction specification is too short."
        )

        return None

    # --------------------------------------------------------
    # Cooldown
    # --------------------------------------------------------

    now = time.time()

    elapsed = (
        now - _last_generation_call
    )

    if elapsed < _generation_cooldown:

        wait = (
            _generation_cooldown
            - elapsed
        )

        print(
            f"⏳ Waiting {wait:.1f}s "
            f"before next image generation..."
        )

        await asyncio.sleep(
            wait
        )

    _last_generation_call = time.time()

    # ========================================================
    # GENERATION PROMPT
    # ========================================================

    final_prompt = f"""
Create a NEW high-resolution sports graphic based on the
reconstruction specification below.

This must be a newly generated image.

============================================================
RECREATION RULES
============================================================

- Create a new independently generated image.
- Recreate the overall visual composition.
- Recreate the visual hierarchy.
- Recreate the general spacing.
- Recreate the card/panel structure.
- Recreate the typography style.
- Recreate the color relationships.
- Recreate the background treatment.
- Recreate the lighting and depth.
- Recreate legitimate visual content described below.

Preserve legitimate visible information as accurately as
possible.

Do NOT invent:
- odds
- scores
- teams
- players
- dates
- times
- picks
- numbers
- other factual information

IMPORTANT:

Do NOT reproduce:
- watermarks
- usernames
- @handles
- channel names
- social-media branding
- third-party logos
- copied brand marks

If the reconstruction specification identifies an area that
previously contained branding, replace that area with a
clean visually appropriate background/design element.

Do not mention the reconstruction process in the image.

Make important text:
- clean
- readable
- sharp
- professionally arranged

The result should look like a polished new sports graphic,
not like an edited screenshot.

============================================================
RECONSTRUCTION SPECIFICATION
============================================================

{description}

============================================================
FINAL QUALITY REQUIREMENTS
============================================================

Generate one cohesive image.

Prioritize:
1. correct composition
2. correct legitimate information
3. readable typography
4. consistent spacing
5. coherent colors
6. professional sports-graphic appearance
7. clean background
8. no watermarks or handles
"""

    try:

        print(
            "🎨 Generating NEW image from "
            "full reconstruction specification..."
        )

        print(
            f"📝 Generation prompt length: "
            f"{len(final_prompt)} characters"
        )

        response = (
            openai_client.images.generate(
                model="gpt-image-2",
                prompt=final_prompt,
                n=1,
            )
        )

        if not response.data:

            print(
                "❌ OpenAI image generation "
                "returned no data."
            )

            return None

        img_data = response.data[0]

        # ----------------------------------------------------
        # Base64 response
        # ----------------------------------------------------

        if getattr(
            img_data,
            "b64_json",
            None,
        ):

            image_bytes = base64.b64decode(
                img_data.b64_json
            )

            print(
                f"✅ New image generated: "
                f"{len(image_bytes)} bytes"
            )

            return BufferedInputFile(
                file=image_bytes,
                filename="regenerated.png",
            )

        # ----------------------------------------------------
        # URL response
        # ----------------------------------------------------

        if getattr(
            img_data,
            "url",
            None,
        ):

            print(
                "✅ New image generated as URL."
            )

            return img_data.url

        print(
            "❌ Image response contained "
            "neither b64_json nor URL."
        )

        return None

    except Exception as e:

        print(
            f"❌ OpenAI image generation error: {e}"
        )

        if hasattr(
            e,
            "response",
        ):

            try:

                print(
                    f"📄 API response: "
                    f"{e.response.text}"
                )

            except Exception:
                pass

        return None


# ============================================================
# COMPLETE IMAGE PIPELINE
# ============================================================

async def regenerate_image_from_bytes(
    image_bytes: bytes,
):
    """
    Complete image recreation pipeline.

        Telegram image/GIF
                ↓
        Prepare image
                ↓
        OpenAI Vision
                ↓
        Detailed deconstruction
                ↓
        OpenAI image generation
                ↓
        NEW clean image
    """

    if not image_bytes:

        print(
            "❌ No image bytes supplied."
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
    # STEP 1
    # ========================================================

    print(
        "1️⃣ STEP 1/2 — "
        "OpenAI Vision deconstruction..."
    )

    description = (
        await describe_image_bytes(
            image_bytes
        )
    )

    if not description:

        print(
            "❌ STEP 1 FAILED — "
            "no reconstruction specification."
        )

        print(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        return None

    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "2️⃣ STEP 2/2 — "
        "OpenAI image generation..."
    )

    generated = (
        await generate_image_from_description(
            description
        )
    )

    if generated:

        print(
            "✅ IMAGE RECREATION PIPELINE COMPLETE"
        )

    else:

        print(
            "❌ STEP 2 FAILED — "
            "image generation failed."
        )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return generated
