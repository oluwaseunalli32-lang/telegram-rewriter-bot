import os
import io
import asyncio
import logging
from collections import defaultdict

from dotenv import load_dotenv

from telethon import TelegramClient
from telethon.tl.types import (
    MessageMediaPhoto,
    DocumentAttributeFilename,
)

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.types import (
    BufferedInputFile,
    InputMediaPhoto,
)

import database
from ai_processor import (
    rewrite_text,
    remove_watermarks_from_bytes,
)

# ============================================================
# LOAD ENVIRONMENT
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
# TELEGRAM CLIENTS
# ============================================================

user_client = TelegramClient(
    "telegram_user_session",
    API_ID,
    API_HASH,
)

# aiogram 3.7+
# Do NOT pass parse_mode directly to Bot().
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(
        parse_mode=None
    ),
)

# ============================================================
# CONSTANTS
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


# ============================================================
# FILE / MEDIA HELPERS
# ============================================================

def get_filename(message):
    """
    Try to recover the original Telegram filename.
    """

    if not message or not message.media:
        return "image.jpg"

    document = getattr(
        message,
        "document",
        None,
    )

    if document:
        attributes = getattr(
            document,
            "attributes",
            None,
        )

        if attributes:
            for attr in attributes:
                if isinstance(
                    attr,
                    DocumentAttributeFilename,
                ):
                    return attr.file_name

    return "image.jpg"


def get_media_type(message):
    """
    Determine how the media should be reposted.
    """

    media = message.media

    # Telegram photo
    if isinstance(
        media,
        MessageMediaPhoto,
    ):
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

        # GIF
        if mime == "image/gif":
            return "animation"

        # Video
        if mime.startswith("video/"):
            return "video"

        # Image/document
        if mime.startswith("image/"):
            return "image"

        # Fallback to filename
        filename = get_filename(
            message
        ).lower()

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

                if ext in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                }:
                    return "image"

    return "document"


async def download_media(message):
    """
    Download Telegram media directly into memory.
    """

    buffer = io.BytesIO()

    await user_client.download_media(
        message,
        file=buffer,
    )

    return buffer.getvalue()


# ============================================================
# WATERMARK PROCESSING
# ============================================================

async def process_image(
    media_bytes,
    filename,
):
    """
    Send still images through the AI watermark remover.

    GIFs/videos are intentionally skipped so the original
    animation/video is preserved and no frame-by-frame
    OpenAI charges occur.
    """

    if not media_bytes:
        return None

    lower_name = (
        filename or ""
    ).lower()

    # Do not send animated/video files to the image editor.
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
            "⏭️ %s is animation/video. "
            "Keeping original.",
            filename,
        )
        return None

    logger.info(
        "🧹 Checking %s for watermark...",
        filename,
    )

    try:
        cleaned = await remove_watermarks_from_bytes(
            media_bytes,
            filename,
        )

        if cleaned:
            logger.info(
                "✅ Watermark-removed image received."
            )
            return cleaned

        logger.info(
            "📌 No edited image returned. "
            "Using original."
        )

        return None

    except Exception:
        logger.exception(
            "❌ Watermark processing failed for %s. "
            "Using original.",
            filename,
        )
        return None


# ============================================================
# CAPTION
# ============================================================

async def prepare_caption(message):
    """
    Preserve caption exactly except for the rules handled
    by rewrite_text():
      - remove *
      - replace @cappersfree
    """

    caption = (
        message.message
        if message.message
        else ""
    )

    if not caption:
        return None

    return await rewrite_text(
        caption
    )


# ============================================================
# SEND SINGLE MESSAGE
# ============================================================

async def send_single(message):
    """
    Download, optionally clean the image, and repost.
    """

    try:
        # ----------------------------------------------------
        # TEXT ONLY
        # ----------------------------------------------------

        if not message.media:

            caption = await prepare_caption(
                message
            )

            if caption:
                await bot.send_message(
                    TARGET_CHANNEL,
                    caption,
                )

            logger.info(
                "✅ Text message %s posted.",
                message.id,
            )

            return True

        # ----------------------------------------------------
        # GET MEDIA INFO
        # ----------------------------------------------------

        filename = get_filename(
            message
        )

        media_type = get_media_type(
            message
        )

        logger.info(
            "📥 Downloading message %s | type=%s | file=%s",
            message.id,
            media_type,
            filename,
        )

        original_bytes = await download_media(
            message
        )

        if not original_bytes:
            logger.error(
                "❌ Failed to download media "
                "for message %s.",
                message.id,
            )
            return False

        # ----------------------------------------------------
        # CAPTION
        # ----------------------------------------------------

        caption = await prepare_caption(
            message
        )

        # ----------------------------------------------------
        # AI WATERMARK REMOVAL
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # PREPARE TELEGRAM FILE
        # ----------------------------------------------------

        file = BufferedInputFile(
            final_bytes,
            filename=filename,
        )

        # ----------------------------------------------------
        # PHOTO / IMAGE
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
        # OTHER DOCUMENT
        # ----------------------------------------------------

        else:

            await bot.send_document(
                TARGET_CHANNEL,
                document=file,
                caption=caption,
            )

        logger.info(
            "✅ Message %s posted successfully.",
            message.id,
        )

        return True

    except Exception:
        logger.exception(
            "❌ Failed processing message %s.",
            message.id,
        )

        return False


# ============================================================
# ALBUM PROCESSING
# ============================================================

async def process_album(messages):
    """
    Process a Telegram album.

    Still images are grouped together where Telegram permits it.
    Videos/GIFs/documents are sent separately.
    """

    if not messages:
        return False

    messages = sorted(
        messages,
        key=lambda m: m.id,
    )

    logger.info(
        "📦 Processing album with %d items.",
        len(messages),
    )

    # --------------------------------------------------------
    # FIND ALBUM CAPTION
    # --------------------------------------------------------

    caption = None

    for msg in messages:
        if msg.message:
            caption = await rewrite_text(
                msg.message
            )
            break

    # --------------------------------------------------------
    # COLLECT PHOTOS
    # --------------------------------------------------------

    photo_items = []

    # --------------------------------------------------------
    # PROCESS EACH ITEM
    # --------------------------------------------------------

    for msg in messages:

        if not msg.media:
            continue

        filename = get_filename(
            msg
        )

        media_type = get_media_type(
            msg
        )

        logger.info(
            "📥 Downloading album item %s | type=%s | file=%s",
            msg.id,
            media_type,
            filename,
        )

        original_bytes = await download_media(
            msg
        )

        if not original_bytes:
            logger.error(
                "❌ Could not download "
                "album item %s.",
                msg.id,
            )
            continue

        # ----------------------------------------------------
        # STILL IMAGE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # GIF / VIDEO / DOCUMENT
        # ----------------------------------------------------

        else:

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

            # Caption should only appear once.
            caption = None

    # --------------------------------------------------------
    # SEND PHOTO ALBUM
    # Telegram allows max 10 items per media group.
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
        "✅ Album processed successfully."
    )

    return True


# ============================================================
# DATABASE START POSITION
# ============================================================

def get_start_message_id():
    """
    Read the last processed message ID from the database.
    """

    try:
        return database.get_last_processed(
            SOURCE_CHANNEL
        )

    except Exception:
        logger.exception(
            "⚠️ Could not read last processed "
            "message from database."
        )
        return None


def save_last_message_id(message_id):
    """
    Save the last processed message ID.
    """

    try:
        database.set_last_processed(
            SOURCE_CHANNEL,
            message_id,
        )

    except Exception:
        logger.exception(
            "⚠️ Could not save last processed "
            "message ID %s.",
            message_id,
        )


# ============================================================
# CHANNEL POLLING
# ============================================================

async def process_channel():
    """
    Poll the source channel for new messages.
    """

    logger.info(
        "📡 Starting source-channel polling..."
    )

    last_message_id = (
        get_start_message_id()
    )

    # --------------------------------------------------------
    # FIRST RUN
    # --------------------------------------------------------

    if last_message_id is None:

        latest = await user_client.get_messages(
            SOURCE_CHANNEL,
            limit=1,
        )

        if latest:

            last_message_id = latest[0].id

            save_last_message_id(
                last_message_id
            )

            logger.info(
                "📌 Initial position set to "
                "message %s. Existing messages "
                "will not be replayed.",
                last_message_id,
            )

    else:

        logger.info(
            "📌 Resuming after message %s.",
            last_message_id,
        )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while True:

        try:

            newest_messages = (
                await user_client.get_messages(
                    SOURCE_CHANNEL,
                    min_id=(
                        last_message_id
                        or 0
                    ),
                    limit=100,
                    reverse=True,
                )
            )

            if newest_messages:

                # ------------------------------------------------
                # GROUP ALBUMS
                # ------------------------------------------------

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
                # NORMAL MESSAGES
                # ------------------------------------------------

                for msg in normal_messages:

                    success = await send_single(
                        msg
                    )

                    if success:

                        last_message_id = max(
                            last_message_id or 0,
                            msg.id,
                        )

                        save_last_message_id(
                            last_message_id
                        )

                # ------------------------------------------------
                # ALBUMS
                # ------------------------------------------------

                sorted_albums = sorted(
                    album_groups.items(),
                    key=lambda item: min(
                        m.id
                        for m in item[1]
                    ),
                )

                for (
                    group_id,
                    group_messages,
                ) in sorted_albums:

                    success = await process_album(
                        group_messages
                    )

                    if success:

                        highest_id = max(
                            m.id
                            for m in group_messages
                        )

                        last_message_id = max(
                            last_message_id or 0,
                            highest_id,
                        )

                        save_last_message_id(
                            last_message_id
                        )

            await asyncio.sleep(
                POLL_INTERVAL
            )

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "❌ Polling loop error."
            )

            await asyncio.sleep(
                POLL_INTERVAL
            )


# ============================================================
# VALIDATION
# ============================================================

def validate_environment():
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

    if not SOURCE_CHANNEL:
        raise RuntimeError(
            "SOURCE_CHANNEL is missing."
        )

    if not TARGET_CHANNEL:
        raise RuntimeError(
            "TARGET_CHANNEL is missing."
        )


# ============================================================
# STARTUP
# ============================================================

async def main():

    validate_environment()

    logger.info(
        "🚀 Starting Telegram Image Recreation Bot..."
    )

    logger.info(
        "📌 Source: %s",
        SOURCE_CHANNEL,
    )

    logger.info(
        "📌 Target: %s",
        TARGET_CHANNEL,
    )

    logger.info(
        "🧹 Watermark removal: ENABLED"
    )

    logger.info(
        "🤖 AI editing is used only for still images."
    )

    logger.info(
        "🎞️ GIFs/videos are preserved unchanged."
    )

    logger.info(
        "📝 Caption handling: "
        "remove '*' + exact username replacement"
    )

    # --------------------------------------------------------
    # OPENAI STATUS
    # --------------------------------------------------------

    if os.getenv(
        "OPENAI_API_KEY",
        "",
    ).strip():

        logger.info(
            "✅ OPENAI_API_KEY detected."
        )

        logger.info(
            "🎨 OpenAI image quality: %s",
            os.getenv(
                "OPENAI_IMAGE_QUALITY",
                "low",
            ),
        )

        logger.info(
            "🤖 OpenAI image model: %s",
            os.getenv(
                "OPENAI_IMAGE_MODEL",
                "gpt-image-2",
            ),
        )

    else:

        logger.warning(
            "⚠️ OPENAI_API_KEY is missing. "
            "Images will be reposted without AI editing."
        )

    # --------------------------------------------------------
    # START TELETHON
    # --------------------------------------------------------

    await user_client.start(
        phone=PHONE_NUMBER
    )

    logger.info(
        "✅ Telegram user client connected!"
    )

    # --------------------------------------------------------
    # START POLLING
    # --------------------------------------------------------

    try:

        await process_channel()

    finally:

        try:
            await user_client.disconnect()
        except Exception:
            pass

        try:
            await bot.session.close()
        except Exception:
            pass


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    try:
        asyncio.run(
            main()
        )

    except KeyboardInterrupt:
        logger.info(
            "🛑 Bot stopped."
        )

    except Exception:
        logger.exception(
            "💥 Fatal startup error."
        )
        raise
