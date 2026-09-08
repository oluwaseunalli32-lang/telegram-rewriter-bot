import os
import io
import asyncio
import logging
from collections import defaultdict

from dotenv import load_dotenv

from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, DocumentAttributeFilename

from aiogram import Bot
from aiogram.types import (
    BufferedInputFile,
    InputMediaPhoto,
)
from aiogram.enums import ParseMode

import database
from ai_processor import (
    rewrite_text,
    remove_watermarks_from_bytes,
)

# ============================================================
# ENV
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "").strip()
PHONE_NUMBER = os.getenv("PHONE_NUMBER", "").strip()

SOURCE_CHANNEL = int(
    os.getenv(
        "SOURCE_CHANNEL",
        "-1003593544389",
    )
)

TARGET_CHANNEL = int(
    os.getenv(
        "TARGET_CHANNEL",
        "-1004415621706",
    )
)

POLL_INTERVAL = int(
    os.getenv(
        "POLL_INTERVAL",
        "5",
    )
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)

# ============================================================
# CLIENTS
# ============================================================

user_client = TelegramClient(
    "telegram_user_session",
    API_ID,
    API_HASH,
)

bot = Bot(
    token=BOT_TOKEN,
    parse_mode=ParseMode.HTML,
)

# ============================================================
# HELPERS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".mp4",
    ".mov",
    ".m4v",
    ".webm",
}


def get_filename(message):
    if not message or not message.media:
        return "image.jpg"

    document = getattr(
        message,
        "document",
        None,
    )

    if document and getattr(
        document,
        "attributes",
        None,
    ):
        for attr in document.attributes:
            if isinstance(
                attr,
                DocumentAttributeFilename,
            ):
                return attr.file_name

    return "image.jpg"


def get_media_type(message):
    media = message.media

    if isinstance(media, MessageMediaPhoto):
        return "photo"

    document = getattr(
        message,
        "document",
        None,
    )

    if document:
        mime = (
            getattr(
                document,
                "mime_type",
                "",
            )
            or ""
        ).lower()

        if mime == "image/gif":
            return "animation"

        if mime.startswith("video/"):
            return "video"

        if mime.startswith("image/"):
            return "image"

        filename = get_filename(message).lower()

        for ext in SUPPORTED_EXTENSIONS:
            if filename.endswith(ext):
                if ext == ".gif":
                    return "animation"

                if ext in {
                    ".mp4",
                    ".mov",
                    ".m4v",
                    ".webm",
                }:
                    return "video"

                return "image"

    return "document"


async def download_media(message):
    buffer = io.BytesIO()

    await user_client.download_media(
        message,
        file=buffer,
    )

    return buffer.getvalue()


# ============================================================
# IMAGE PROCESSING
# ============================================================

async def process_image(
    media_bytes: bytes,
    filename: str,
):
    """
    Only raster images go through OpenAI.

    GIF/video are left untouched to avoid:
    - frame-by-frame API costs
    - breaking animation
    - unnecessary quality loss
    """

    if not media_bytes:
        return None

    lower_name = filename.lower()

    if lower_name.endswith(
        (
            ".gif",
            ".mp4",
            ".mov",
            ".m4v",
            ".webm",
        )
    ):
        logger.info(
            "⏭️ Animation/video detected. "
            "Keeping original media."
        )
        return None

    logger.info(
        "🧹 Checking image for watermark: %s",
        filename,
    )

    cleaned = await remove_watermarks_from_bytes(
        media_bytes,
        filename,
    )

    if cleaned:
        logger.info(
            "✅ Cleaned image returned."
        )
        return cleaned

    logger.info(
        "📌 No AI edit needed. Using original."
    )

    return None


# ============================================================
# CAPTION
# ============================================================

async def prepare_caption(message):
    caption = message.message or ""

    if not caption:
        return None

    return await rewrite_text(
        caption
    )


# ============================================================
# SEND SINGLE MESSAGE
# ============================================================

async def send_single(message):
    try:
        if not message.media:
            caption = await prepare_caption(
                message
            )

            if caption:
                await bot.send_message(
                    TARGET_CHANNEL,
                    caption,
                )

            return True

        filename = get_filename(
            message
        )

        media_type = get_media_type(
            message
        )

        original_bytes = await download_media(
            message
        )

        if not original_bytes:
            logger.error(
                "❌ Failed to download media."
            )
            return False

        caption = await prepare_caption(
            message
        )

        cleaned_bytes = None

        if media_type in {
            "photo",
            "image",
        }:
            cleaned_bytes = await process_image(
                original_bytes,
                filename,
            )

        final_bytes = (
            cleaned_bytes
            if cleaned_bytes
            else original_bytes
        )

        file = BufferedInputFile(
            final_bytes,
            filename=filename,
        )

        # ----------------------------------------------------
        # PHOTO
        # ----------------------------------------------------

        if media_type in {
            "photo",
            "image",
        }:
            await bot.send_photo(
                TARGET_CHANNEL,
                photo=file,
                caption=caption,
            )

        # ----------------------------------------------------
        # GIF
        # ----------------------------------------------------

        elif media_type == "animation":
            await bot.send_animation(
                TARGET_CHANNEL,
                animation=file,
                caption=caption,
            )

        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        elif media_type == "video":
            await bot.send_video(
                TARGET_CHANNEL,
                video=file,
                caption=caption,
            )

        # ----------------------------------------------------
        # DOCUMENT
        # ----------------------------------------------------

        else:
            await bot.send_document(
                TARGET_CHANNEL,
                document=file,
                caption=caption,
            )

        logger.info(
            "✅ Message %s posted.",
            message.id,
        )

        return True

    except Exception:
        logger.exception(
            "❌ Failed processing message %s",
            message.id,
        )

        return False


# ============================================================
# ALBUM
# ============================================================

async def process_album(messages):
    if not messages:
        return False

    logger.info(
        "📦 Processing album with %d items.",
        len(messages),
    )

    messages = sorted(
        messages,
        key=lambda m: m.id,
    )

    # Telegram captions generally belong to one item.
    caption = None

    for msg in messages:
        if msg.message:
            caption = await rewrite_text(
                msg.message
            )
            break

    photo_items = []

    for msg in messages:

        if not msg.media:
            continue

        filename = get_filename(
            msg
        )

        media_type = get_media_type(
            msg
        )

        original_bytes = await download_media(
            msg
        )

        if not original_bytes:
            logger.error(
                "❌ Album item %s "
                "could not be downloaded.",
                msg.id,
            )
            continue

        # Only process still images.
        if media_type in {
            "photo",
            "image",
        }:
            cleaned = await process_image(
                original_bytes,
                filename,
            )

            final_bytes = (
                cleaned
                if cleaned
                else original_bytes
            )

            photo_items.append(
                (
                    final_bytes,
                    filename,
                )
            )

        else:
            # GIF/video/document cannot safely be merged
            # into a Telegram photo album.
            file = BufferedInputFile(
                original_bytes,
                filename=filename,
            )

            if media_type == "animation":

                await bot.send_animation(
                    TARGET_CHANNEL,
                    animation=file,
                    caption=caption,
                )

            elif media_type == "video":

                await bot.send_video(
                    TARGET_CHANNEL,
                    video=file,
                    caption=caption,
                )

            else:

                await bot.send_document(
                    TARGET_CHANNEL,
                    document=file,
                    caption=caption,
                )

            caption = None

    # --------------------------------------------------------
    # SEND PHOTO ALBUM IN TELEGRAM'S CHUNKS OF 10
    # --------------------------------------------------------

    if photo_items:

        for start in range(
            0,
            len(photo_items),
            10,
        ):

            chunk = photo_items[
                start:start + 10
            ]

            media_group = []

            for index, (
                image_bytes,
                filename,
            ) in enumerate(chunk):

                file = BufferedInputFile(
                    image_bytes,
                    filename=filename,
                )

                media_group.append(
                    InputMediaPhoto(
                        media=file,
                        caption=(
                            caption
                            if index == 0
                            else None
                        ),
                    )
                )

            await bot.send_media_group(
                TARGET_CHANNEL,
                media=media_group,
            )

            caption = None

    logger.info(
        "✅ Album processed."
    )

    return True


# ============================================================
# CHANNEL POLLING
# ============================================================

async def process_channel():
    logger.info(
        "📡 Starting source-channel polling..."
    )

    last_message_id = database.get_last_processed(
        SOURCE_CHANNEL
    )

    if last_message_id is None:
        latest = await user_client.get_messages(
            SOURCE_CHANNEL,
            limit=1,
        )

        if latest:
            last_message_id = latest[0].id

            database.set_last_processed(
                SOURCE_CHANNEL,
                last_message_id,
            )

        logger.info(
            "📌 Starting from message ID %s",
            last_message_id,
        )

    while True:

        try:

            newest_messages = await user_client.get_messages(
                SOURCE_CHANNEL,
                min_id=last_message_id or 0,
                limit=100,
                reverse=True,
            )

            if newest_messages:

                album_groups = defaultdict(list)
                normal_messages = []

                for msg in newest_messages:

                    if msg.grouped_id:
                        album_groups[
                            msg.grouped_id
                        ].append(msg)

                    else:
                        normal_messages.append(
                            msg
                        )

                # ------------------------------------------------
                # PROCESS NORMAL MESSAGES
                # ------------------------------------------------

                for msg in normal_messages:

                    await send_single(
                        msg
                    )

                    last_message_id = max(
                        last_message_id or 0,
                        msg.id,
                    )

                    database.set_last_processed(
                        SOURCE_CHANNEL,
                        last_message_id,
                    )

                # ------------------------------------------------
                # PROCESS ALBUMS
                # ------------------------------------------------

                for group_id, group in sorted(
                    album_groups.items(),
                    key=lambda item: min(
                        m.id
                        for m in item[1]
                    ),
                ):

                    await process_album(
                        group
                    )

                    last_message_id = max(
                        last_message_id or 0,
                        max(
                            m.id
                            for m in group
                        ),
                    )

                    database.set_last_processed(
                        SOURCE_CHANNEL,
                        last_message_id,
                    )

            await asyncio.sleep(
                POLL_INTERVAL
            )

        except Exception:
            logger.exception(
                "❌ Polling loop error."
            )

            await asyncio.sleep(
                POLL_INTERVAL
            )


# ============================================================
# STARTUP
# ============================================================

async def main():
    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is missing."
        )

    if not API_ID:
        raise RuntimeError(
            "API_ID is missing."
        )

    if not API_HASH:
        raise RuntimeError(
            "API_HASH is missing."
        )

    if not PHONE_NUMBER:
        raise RuntimeError(
            "PHONE_NUMBER is missing."
        )

    if not os.getenv(
        "OPENAI_API_KEY"
    ):
        logger.warning(
            "⚠️ OPENAI_API_KEY is missing. "
            "Bot will repost originals."
        )

    logger.info(
        "🚀 Starting Telegram Image Bot..."
    )

    logger.info(
        "🧹 Watermark removal: AI edit mode"
    )

    logger.info(
        "💰 OpenAI quality: %s",
        os.getenv(
            "OPENAI_IMAGE_QUALITY",
            "low",
        ),
    )

    logger.info(
        "📝 Caption handling: "
        "remove * + exact username replacement"
    )

    await user_client.start(
        phone=PHONE_NUMBER
    )

    logger.info(
        "✅ Telegram user client connected."
    )

    await process_channel()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    asyncio.run(
        main()
    )
