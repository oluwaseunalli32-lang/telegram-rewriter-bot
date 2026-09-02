import os
import re
import logging


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(
    "ai_processor"
)


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


if (
    NEW_MENTION
    and not NEW_MENTION.startswith("@")
):

    NEW_MENTION = (
        "@"
        + NEW_MENTION
    )


# ============================================================
# STARTUP
# ============================================================

logger.info(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)

logger.info(
    "📝 CAPTION PROCESSOR STARTED"
)

logger.info(
    "👤 OLD MENTION: %s",
    OLD_MENTION,
)

logger.info(
    "👤 NEW MENTION: %s",
    NEW_MENTION,
)

logger.info(
    "🤖 OpenAI: DISABLED"
)

logger.info(
    "🧹 Watermark removal: DISABLED"
)

logger.info(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)


# ============================================================
# CAPTION PROCESSING
# ============================================================

def replace_username(
    text: str,
) -> str:

    """
    ONLY:

    1. Remove literal *
    2. Replace OLD_MENTION with NEW_MENTION

    Nothing else.
    """

    if not text:

        return text

    result = text.replace(
        "*",
        "",
    )

    if NEW_MENTION:

        result = re.sub(
            re.escape(
                OLD_MENTION
            ),
            NEW_MENTION,
            result,
            flags=re.IGNORECASE,
        )

    return result


# ============================================================
# MAIN CAPTION FUNCTION
# ============================================================

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
        "📝 ORIGINAL: %r",
        original_text,
    )

    logger.info(
        "📝 FINAL:    %r",
        result,
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return result


# ============================================================
# DISABLED AI FUNCTION
# ============================================================

async def remove_watermarks_from_bytes(
    image_bytes: bytes,
    filename: str = "",
    mime_type: str = "",
):
    """
    Deliberately disabled.

    The bot currently reposts the original media unchanged.
    """

    logger.info(
        "⏸️ Watermark removal is currently disabled."
    )

    return None


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

async def regenerate_image_from_bytes(
    image_bytes: bytes,
):

    logger.info(
        "⏸️ Image regeneration is currently disabled."
    )

    return None
