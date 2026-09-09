import os
import io
import asyncio
import logging
from collections import defaultdict

from dotenv import load_dotenv

from telethon import TelegramClient
from telethon.sessions import StringSession
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

API_ID_RAW = os.getenv("API_ID", "").strip()
API_HASH = os.getenv("API_HASH", "").strip()
PHONE_NUMBER = os.getenv("PHONE_NUMBER", "").strip()

# IMPORTANT:
# This is the Telethon StringSession generated once locally.
TELEGRAM_SESSION = os.getenv(
    "TELEGRAM_SESSION",
    "",
).strip()

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
# VALIDATE API ID
# ============================================================

try:
    API_ID = int(API_ID_RAW)
except ValueError:
    API_ID = 0

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
# TELEGRAM USER CLIENT
# ============================================================

if TELEGRAM_SESSION:
    logger.info(
        "🔐 Using Telegram StringSession."
    )

    user_client = TelegramClient(
        StringSession(TELEGRAM_SESSION),
        API_ID,
        API_HASH,
    )

else:
    logger.error(
        "❌ TELEGRAM_SESSION is missing."
    )

    user_client = TelegramClient(
        StringSession(),
        API_ID,
        API_HASH,
    )

# ============================================================
# AIROGRAM BOT
# ============================================================

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(
        parse_mode=None,
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
# ENV VALIDATION
# ============================================================

def validate_environment():
    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is missing."
        )

    if not API_ID:
        raise RuntimeError(
            "API_ID is missing or invalid."
        )

    if not API_HASH:
        raise RuntimeError(
            "API_HASH is missing."
        )

    if not TELEGRAM_SESSION:
        raise RuntimeError(
            "TELEGRAM_SESSION is missing. "
            "Generate a Telethon StringSession locally "
            "and add it to Render Environment Variables."
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
# FILE / MEDIA HELPERS
# ============================================================

def get_filename(message):
    """
    Get the original Telegram filename where available.
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
    Determine how Telegram media should be reposted.
    """

    media = message.media

    # Native Telegram photo
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

        # Image
        if mime.startswith("image/"):
            return "image"

        # Filename fallback
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
    Download Telegram media into memory.
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
    Send still images to the AI watermark-removal processor.

    GIFs and videos are left untouched to avoid expensive
    frame-by-frame processing.
    """

    if not media_bytes:
        return None

    lower_name = (
        filename or ""
    ).lower()

    # Preserve animation/video exactly.
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

        cleaned = (
            await remove_watermarks_from_bytes(
                media_bytes,
                filename,
            )
        )

        if cleaned:

            logger.info(
                "✅ Cleaned image received."
            )

            return cleaned

        logger.info(
            "📌 Original image retained."
        )

        return None

    except Exception:

        logger.exception(
            "❌ Watermark processing failed "
            "for %s. Using original.",
            filename,
        )

        return None


# ============================================================
# CAPTION
# ============================================================

async def prepare_caption(message):
    """
    Caption handling remains:
      1. Remove *
      2. Replace @cappersfree
      3. Change nothing else
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

    try:

        # ----------------------------------------------------
        # TEXT-ONLY MESSAGE
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
        # MEDIA INFO
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
        # WATERMARK REMOVAL
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
        # TELEGRAM FILE
        # ----------------------------------------------------

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
    # PHOTO ITEMS
    # --------------------------------------------------------

    photo_items = []

    # --------------------------------------------------------
    # PROCESS ALBUM ITEMS
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
                "❌ Could not download album item %s.",
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
        # VIDEO / GIF / DOCUMENT
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

            caption = None

    # --------------------------------------------------------
    # SEND PHOTO GROUPS
    # Telegram maximum is 10 photos per media group.
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
# DATABASE
# ============================================================

def get_start_message_id():

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
# SOURCE CHANNEL POLLING
# ============================================================

async def process_channel():

    logger.info(
        "📡 Starting source-channel polling..."
    )

    last_message_id = (
        get_start_message_id()
    )

    # --------------------------------------------------------
    # INITIAL START POSITION
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
                "📌 Initial position set to message %s.",
                last_message_id,
            )

    else:

        logger.info(
            "📌 Resuming after message %s.",
            last_message_id,
        )

    # --------------------------------------------------------
    # POLLING LOOP
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

                album_groups = defaultdict(list)

                normal_messages = []

                # ------------------------------------------------
                # GROUP ALBUMS
                # ------------------------------------------------

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
        "🔐 Telegram authentication: StringSession"
    )

    logger.info(
        "🧹 Watermark removal: ENABLED"
    )

    logger.info(
        "🤖 AI editing: still images only"
    )

    logger.info(
        "🎞️ GIFs/videos: original media preserved"
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
            "Images will be posted without AI editing."
        )

    # --------------------------------------------------------
    # CONNECT TELEGRAM
    # --------------------------------------------------------

    logger.info(
        "🔌 Connecting to Telegram..."
    )

    await user_client.connect()

    # --------------------------------------------------------
    # CHECK AUTHENTICATION
    # --------------------------------------------------------

    try:

        authorized = await user_client.is_user_authorized()

    except Exception:

        logger.exception(
            "❌ Could not verify Telegram authorization."
        )

        await user_client.disconnect()

        raise

    if not authorized:

        await user_client.disconnect()

        raise RuntimeError(
            "Telegram StringSession is not authorized. "
            "Generate a new StringSession locally and "
            "add it to TELEGRAM_SESSION on Render."
        )

    # --------------------------------------------------------
    # GET ACCOUNT
    # --------------------------------------------------------

    try:

        me = await user_client.get_me()

        if me:

            username = (
                f"@{me.username}"
                if me.username
                else "(no username)"
            )

            logger.info(
                "✅ Telegram account authenticated: %s | %s",
                me.first_name or "",
                username,
            )

    except Exception:

        logger.warning(
            "⚠️ Telegram connected, but account "
            "information could not be read."
        )

    logger.info(
        "✅ Telegram user client connected!"
    )

    # --------------------------------------------------------
    # START PROCESSING
    # --------------------------------------------------------

    try:

        await process_channel()

    finally:

        logger.info(
            "🛑 Shutting down..."
        )

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
