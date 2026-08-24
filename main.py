import os
import re
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from aiogram import Bot
from aiogram.types import (
    BufferedInputFile,
    InputMediaPhoto,
    InputMediaVideo,
)

from telethon import TelegramClient, errors


# ============================================================
# LOAD ENVIRONMENT FIRST
# ============================================================

env_path = (
    Path(__file__).parent
    / ".env"
)

load_dotenv(
    dotenv_path=env_path
)


# ============================================================
# IMPORTS THAT USE ENVIRONMENT
# ============================================================

import database

from ai_processor import (
    rewrite_text,
    remove_watermarks_from_bytes,
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

logger = logging.getLogger(
    __name__
)


# ============================================================
# ENVIRONMENT
# ============================================================

BOT_TOKEN = os.getenv(
    "BOT_TOKEN"
)

API_ID = (
    int(os.getenv("API_ID"))
    if os.getenv("API_ID")
    else 0
)

API_HASH = os.getenv(
    "API_HASH"
)

PHONE = os.getenv(
    "PHONE_NUMBER"
)


if (
    not BOT_TOKEN
    or not API_ID
    or not API_HASH
    or not PHONE
):

    logger.error(
        "❌ Missing environment variables."
    )

    raise SystemExit(1)


# ============================================================
# TELEGRAM CLIENTS
# ============================================================

bot = Bot(
    token=BOT_TOKEN
)

user_client = TelegramClient(
    "session_name",
    API_ID,
    API_HASH,
)


# ============================================================
# CHANNEL CONFIGURATION
# ============================================================

SOURCE_CHANNEL_ID = -1003593544389

TARGET_CHANNEL_ID = -1004415621706


# ============================================================
# ALBUM CONFIGURATION
# ============================================================

# Wait for Telegram to finish delivering the album.
ALBUM_SETTLE_SECONDS = float(
    os.getenv(
        "ALBUM_SETTLE_SECONDS",
        "2.5",
    )
)

# Telegram media groups support up to 10 items.
TELEGRAM_ALBUM_SIZE = 10


# ============================================================
# PROCESSING STATE
# ============================================================

processing_ids = set()

processing_lock = asyncio.Lock()

last_processed = {}


# ============================================================
# CONNECTION
# ============================================================

async def ensure_connection():

    if user_client.is_connected():
        return True

    logger.warning(
        "⚠️ Telegram client disconnected. Reconnecting..."
    )

    try:

        await user_client.connect()

        if not user_client.is_connected():

            await user_client.start(
                phone=PHONE
            )

        logger.info(
            "✅ Reconnected successfully."
        )

        return True

    except Exception as e:

        logger.exception(
            "❌ Reconnection failed: %s",
            e,
        )

        return False


# ============================================================
# MEDIA DETECTION
# ============================================================

def get_image_media(msg):
    """
    Return Telegram media that can be cleaned.

    Supports:
        photos
        images
        GIFs
        MP4 GIFs
        MOV
        WebM
    """

    # --------------------------------------------------------
    # Normal Telegram photo.
    # --------------------------------------------------------

    if getattr(
        msg,
        "photo",
        None,
    ):

        return msg.photo

    # --------------------------------------------------------
    # Document media.
    # --------------------------------------------------------

    document = getattr(
        msg,
        "document",
        None,
    )

    if document:

        mime_type = (
            getattr(
                document,
                "mime_type",
                "",
            )
            or ""
        ).lower()

        if mime_type.startswith(
            "image/"
        ):

            return document

        if mime_type.startswith(
            "video/"
        ):

            return document

        attributes = (
            getattr(
                document,
                "attributes",
                None,
            )
            or []
        )

        supported_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
            ".mp4",
            ".mov",
            ".m4v",
            ".webm",
        }

        for attribute in attributes:

            filename = (
                getattr(
                    attribute,
                    "file_name",
                    "",
                )
                or ""
            )

            extension = (
                Path(filename)
                .suffix
                .lower()
            )

            if extension in supported_extensions:
                return document

    # --------------------------------------------------------
    # Telethon fallback.
    # --------------------------------------------------------

    media = getattr(
        msg,
        "media",
        None,
    )

    if (
        media
        and hasattr(
            media,
            "photo",
        )
    ):

        return media

    return None


# ============================================================
# CAPTION EXTRACTION
# ============================================================

def get_message_text(msg):

    return (
        getattr(
            msg,
            "text",
            None,
        )
        or getattr(
            msg,
            "message",
            None,
        )
        or getattr(
            msg,
            "caption",
            None,
        )
        or ""
    )


# ============================================================
# CLEAN ONE TELEGRAM MESSAGE
# ============================================================

async def clean_message_media(
    msg,
):
    """
    Download one Telegram media object and clean it.
    """

    media = get_image_media(
        msg
    )

    if not media:

        logger.error(
            "❌ [%s] No supported media.",
            msg.id,
        )

        return None

    logger.info(
        "⬇️ [%s] Downloading media...",
        msg.id,
    )

    original_bytes = (
        await user_client.download_media(
            media,
            bytes,
        )
    )

    if not original_bytes:

        logger.error(
            "❌ [%s] Download failed.",
            msg.id,
        )

        return None

    logger.info(
        "📦 [%s] Downloaded %d bytes.",
        msg.id,
        len(original_bytes),
    )

    logger.info(
        "🧹 [%s] Removing CF watermark...",
        msg.id,
    )

    cleaned = (
        await remove_watermarks_from_bytes(
            original_bytes
        )
    )

    if not cleaned:

        logger.error(
            "❌ [%s] Watermark removal failed.",
            msg.id,
        )

        return None

    cleaned_bytes, media_type = cleaned

    logger.info(
        "✅ [%s] Cleaned media: %d bytes (%s)",
        msg.id,
        len(cleaned_bytes),
        media_type,
    )

    return {
        "message_id": msg.id,
        "bytes": cleaned_bytes,
        "type": media_type,
    }


# ============================================================
# PROCESS SINGLE MESSAGE
# ============================================================

async def process_single_message(
    msg,
    target_id,
):
    """
    Process a normal non-album message.
    """

    original_text = get_message_text(
        msg
    )

    # --------------------------------------------------------
    # CAPTION.
    #
    # This remains completely separate from image processing.
    # --------------------------------------------------------

    caption = await rewrite_text(
        original_text
    )

    media = get_image_media(
        msg
    )

    # --------------------------------------------------------
    # MEDIA.
    # --------------------------------------------------------

    if media:

        cleaned = (
            await clean_message_media(
                msg
            )
        )

        if not cleaned:
            return False

        message_id = cleaned[
            "message_id"
        ]

        cleaned_bytes = cleaned[
            "bytes"
        ]

        media_type = cleaned[
            "type"
        ]

        filename_base = (
            f"cleaned_{message_id}"
        )

        # ----------------------------------------------------
        # MP4/video.
        # ----------------------------------------------------

        if media_type == "video":

            file = BufferedInputFile(
                cleaned_bytes,
                filename_base
                + ".mp4",
            )

            await bot.send_video(
                chat_id=target_id,
                video=file,
                caption=(
                    caption[:1024]
                    if caption
                    else None
                ),
                supports_streaming=True,
            )

        # ----------------------------------------------------
        # Actual GIF.
        #
        # Telegram's animation endpoint is used because an
        # actual GIF should remain animated.
        # ----------------------------------------------------

        elif media_type == "gif":

            file = BufferedInputFile(
                cleaned_bytes,
                filename_base
                + ".gif",
            )

            await bot.send_animation(
                chat_id=target_id,
                animation=file,
                caption=(
                    caption[:1024]
                    if caption
                    else None
                ),
            )

        # ----------------------------------------------------
        # Normal image.
        # ----------------------------------------------------

        else:

            file = BufferedInputFile(
                cleaned_bytes,
                filename_base
                + ".png",
            )

            await bot.send_photo(
                chat_id=target_id,
                photo=file,
                caption=(
                    caption[:1024]
                    if caption
                    else None
                ),
            )

        logger.info(
            "✅ [%s] Cleaned media posted.",
            msg.id,
        )

        return True

    # --------------------------------------------------------
    # TEXT ONLY.
    # --------------------------------------------------------

    if caption:

        await bot.send_message(
            chat_id=target_id,
            text=caption,
        )

        logger.info(
            "✅ [%s] Text message posted.",
            msg.id,
        )

        return True

    return True


# ============================================================
# COLLECT COMPLETE TELEGRAM ALBUM
# ============================================================

async def collect_album(
    channel,
    grouped_id,
    minimum_id,
):
    """
    Wait briefly and re-fetch recent messages so all images
    belonging to the Telegram album are collected.
    """

    logger.info(
        "⏳ Waiting %.1fs for album %s...",
        ALBUM_SETTLE_SECONDS,
        grouped_id,
    )

    await asyncio.sleep(
        ALBUM_SETTLE_SECONDS
    )

    album_messages = {}

    async for msg in user_client.iter_messages(
        channel,
        limit=100,
    ):

        if msg.id <= minimum_id:
            break

        if (
            getattr(
                msg,
                "grouped_id",
                None,
            )
            == grouped_id
        ):

            album_messages[
                msg.id
            ] = msg

    messages = list(
        album_messages.values()
    )

    messages.sort(
        key=lambda m: m.id
    )

    logger.info(
        "🖼️ Album %s contains %d item(s).",
        grouped_id,
        len(messages),
    )

    return messages


# ============================================================
# PROCESS TELEGRAM ALBUM
# ============================================================

async def process_album(
    messages,
    target_id,
):
    """
    Clean every item in the album first.

    Nothing is posted until all processable album items
    have been cleaned successfully.
    """

    if not messages:
        return False

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    logger.info(
        "🖼️ ALBUM PROCESSING START"
    )

    logger.info(
        "🖼️ Album size: %d",
        len(messages),
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    # --------------------------------------------------------
    # Find album caption.
    #
    # Telegram normally puts it on the first item.
    # --------------------------------------------------------

    album_caption = ""

    for msg in messages:

        text = get_message_text(
            msg
        )

        if text:

            album_caption = (
                await rewrite_text(
                    text
                )
            )

            break

    # --------------------------------------------------------
    # Clean all album media.
    # --------------------------------------------------------

    cleaned_items = []

    for msg in messages:

        media = get_image_media(
            msg
        )

        if not media:

            logger.warning(
                "⚠️ [%s] Album item has no supported media.",
                msg.id,
            )

            continue

        cleaned = (
            await clean_message_media(
                msg
            )
        )

        if not cleaned:

            logger.error(
                "❌ Album failed because item %s "
                "could not be cleaned.",
                msg.id,
            )

            return False

        cleaned_items.append(
            cleaned
        )

    if not cleaned_items:

        logger.error(
            "❌ Album contains no processable media."
        )

        return False

    # --------------------------------------------------------
    # Albums containing actual GIFs/animations cannot be sent
    # inside Telegram's normal photo/video media group.
    #
    # We therefore:
    #
    #   1. Send normal photo/video items together.
    #   2. Send actual GIF animations separately.
    #
    # Everything is still processed and posted.
    # --------------------------------------------------------

    normal_items = [
        item
        for item in cleaned_items
        if item["type"]
        in {
            "photo",
            "video",
        }
    ]

    gif_items = [
        item
        for item in cleaned_items
        if item["type"] == "gif"
    ]

    # --------------------------------------------------------
    # Send normal media in Telegram groups of <= 10.
    # --------------------------------------------------------

    chunks = [
        normal_items[
            i:i + TELEGRAM_ALBUM_SIZE
        ]
        for i in range(
            0,
            len(normal_items),
            TELEGRAM_ALBUM_SIZE,
        )
    ]

    caption_used = False

    for chunk_index, chunk in enumerate(
        chunks
    ):

        if not chunk:
            continue

        media_group = []

        for item in chunk:

            filename_base = (
                f"album_"
                f"{messages[0].id}_"
                f"{item['message_id']}"
            )

            caption = None

            if (
                not caption_used
                and album_caption
            ):

                caption = (
                    album_caption[:1024]
                )

            if item["type"] == "video":

                file = BufferedInputFile(
                    item["bytes"],
                    filename_base
                    + ".mp4",
                )

                media_group.append(
                    InputMediaVideo(
                        media=file,
                        caption=caption,
                    )
                )

            else:

                file = BufferedInputFile(
                    item["bytes"],
                    filename_base
                    + ".png",
                )

                media_group.append(
                    InputMediaPhoto(
                        media=file,
                        caption=caption,
                    )
                )

            if caption:
                caption_used = True

        logger.info(
            "📤 Sending album chunk %d/%d (%d items)...",
            chunk_index + 1,
            len(chunks),
            len(media_group),
        )

        await bot.send_media_group(
            chat_id=target_id,
            media=media_group,
        )

    # --------------------------------------------------------
    # Send actual GIF animations.
    # --------------------------------------------------------

    for item in gif_items:

        filename_base = (
            f"album_"
            f"{messages[0].id}_"
            f"{item['message_id']}"
        )

        file = BufferedInputFile(
            item["bytes"],
            filename_base
            + ".gif",
        )

        caption = None

        if (
            not caption_used
            and album_caption
        ):

            caption = (
                album_caption[:1024]
            )

            caption_used = True

        logger.info(
            "📤 Sending GIF %s...",
            item["message_id"],
        )

        await bot.send_animation(
            chat_id=target_id,
            animation=file,
            caption=caption,
        )

    logger.info(
        "✅ ALBUM POSTED SUCCESSFULLY"
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return True


# ============================================================
# PROCESS CHANNEL
# ============================================================

async def process_channel(
    source_id,
    target_id,
):

    if not await ensure_connection():

        logger.error(
            "❌ Telegram client disconnected."
        )

        return

    try:

        channel = await user_client.get_entity(
            source_id
        )

        last_id = last_processed.get(
            source_id,
            0,
        )

        # ----------------------------------------------------
        # Fetch new messages chronologically.
        # ----------------------------------------------------

        new_messages = []

        async for msg in user_client.iter_messages(
            channel,
            min_id=last_id,
            reverse=True,
        ):

            new_messages.append(
                msg
            )

        if not new_messages:
            return

        # ----------------------------------------------------
        # Group album messages.
        # ----------------------------------------------------

        groups = {}

        for msg in new_messages:

            grouped_id = getattr(
                msg,
                "grouped_id",
                None,
            )

            if grouped_id:

                key = (
                    "album",
                    grouped_id,
                )

            else:

                key = (
                    "single",
                    msg.id,
                )

            groups.setdefault(
                key,
                [],
            ).append(msg)

        # ----------------------------------------------------
        # Chronological groups.
        # ----------------------------------------------------

        ordered_groups = sorted(
            groups.values(),
            key=lambda group: min(
                msg.id
                for msg in group
            ),
        )

        # ----------------------------------------------------
        # Process each group.
        # ----------------------------------------------------

        for initial_group in ordered_groups:

            first_message = (
                initial_group[0]
            )

            grouped_id = getattr(
                first_message,
                "grouped_id",
                None,
            )

            # =================================================
            # ALBUM
            # =================================================

            if grouped_id:

                messages = (
                    await collect_album(
                        channel,
                        grouped_id,
                        last_id,
                    )
                )

                # If re-fetch failed to find the album,
                # use the messages already fetched.
                if not messages:

                    messages = sorted(
                        initial_group,
                        key=lambda m: m.id,
                    )

                message_keys = [
                    (
                        source_id,
                        msg.id,
                    )
                    for msg in messages
                ]

                async with processing_lock:

                    if any(
                        key in processing_ids
                        for key in message_keys
                    ):

                        logger.warning(
                            "⚠️ Album already processing. "
                            "Skipping duplicate."
                        )

                        continue

                    processing_ids.update(
                        message_keys
                    )

                try:

                    success = (
                        await process_album(
                            messages,
                            target_id,
                        )
                    )

                    if not success:

                        logger.error(
                            "❌ Album failed. "
                            "It will be retried."
                        )

                        continue

                    highest_id = max(
                        msg.id
                        for msg in messages
                    )

                    last_processed[
                        source_id
                    ] = max(
                        last_processed.get(
                            source_id,
                            0,
                        ),
                        highest_id,
                    )

                finally:

                    async with processing_lock:

                        for key in message_keys:

                            processing_ids.discard(
                                key
                            )

            # =================================================
            # SINGLE MESSAGE
            # =================================================

            else:

                msg = first_message

                processing_key = (
                    source_id,
                    msg.id,
                )

                async with processing_lock:

                    if (
                        processing_key
                        in processing_ids
                    ):

                        continue

                    processing_ids.add(
                        processing_key
                    )

                try:

                    logger.info(
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    )

                    logger.info(
                        "📩 Processing message %s",
                        msg.id,
                    )

                    success = (
                        await process_single_message(
                            msg,
                            target_id,
                        )
                    )

                    if not success:

                        logger.error(
                            "❌ Message %s failed. "
                            "It will be retried.",
                            msg.id,
                        )

                        continue

                    last_processed[
                        source_id
                    ] = max(
                        last_processed.get(
                            source_id,
                            0,
                        ),
                        msg.id,
                    )

                    logger.info(
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                    )

                finally:

                    async with processing_lock:

                        processing_ids.discard(
                            processing_key
                        )

            # ------------------------------------------------
            # Small delay between groups.
            # ------------------------------------------------

            await asyncio.sleep(2)

    except errors.rpcerrorlist.AuthKeyError as e:

        logger.error(
            "❌ Telegram authentication error: %s",
            e,
        )

        try:

            await user_client.start(
                phone=PHONE
            )

        except Exception as restart_error:

            logger.error(
                "❌ Could not restart Telegram client: %s",
                restart_error,
            )

    except Exception as e:

        logger.exception(
            "❌ Error processing channel %s: %s",
            source_id,
            e,
        )


# ============================================================
# POLLING
# ============================================================

async def poll_channels():

    while True:

        if not await ensure_connection():

            await asyncio.sleep(10)

            continue

        clients = (
            database.get_all_clients()
        )

        if not clients:

            await asyncio.sleep(10)

            continue

        for client in clients:

            await process_channel(
                client["source"],
                client["target"],
            )

        await asyncio.sleep(5)


# ============================================================
# MAIN
# ============================================================

async def main():

    logger.info(
        "🚀 Starting Telegram watermark-removal bot..."
    )

    logger.info(
        "   Caption: existing exact replacement logic"
    )

    logger.info(
        "   Image: OCR + temporal CF detection + OpenCV inpainting"
    )

    logger.info(
        "   OpenAI image generation: DISABLED"
    )

    logger.info(
        "   GIF pulsing-mask learning: ENABLED"
    )

    logger.info(
        "   Telegram albums: ENABLED"
    )

    # --------------------------------------------------------
    # Start Telethon.
    # --------------------------------------------------------

    await user_client.start(
        phone=PHONE
    )

    if not user_client.is_connected():

        logger.error(
            "❌ Failed to connect Telegram client."
        )

        return

    logger.info(
        "✅ User client connected."
    )

    # --------------------------------------------------------
    # Register source → target.
    # --------------------------------------------------------

    existing = (
        database.get_target_for_source(
            SOURCE_CHANNEL_ID
        )
    )

    if existing is None:

        logger.info(
            "📝 Adding client: %s → %s",
            SOURCE_CHANNEL_ID,
            TARGET_CHANNEL_ID,
        )

        database.add_client(
            SOURCE_CHANNEL_ID,
            TARGET_CHANNEL_ID,
        )

    else:

        logger.info(
            "✅ Client already registered: %s → %s",
            SOURCE_CHANNEL_ID,
            existing,
        )

    # --------------------------------------------------------
    # Start from newest message.
    #
    # This prevents old posts from being processed after a
    # fresh deployment.
    # --------------------------------------------------------

    for client in (
        database.get_all_clients()
    ):

        source = client[
            "source"
        ]

        try:

            channel = await user_client.get_entity(
                source
            )

            async for msg in user_client.iter_messages(
                channel,
                limit=1,
            ):

                last_processed[
                    source
                ] = msg.id

                logger.info(
                    "📌 Starting after message %s "
                    "for channel %s",
                    msg.id,
                    source,
                )

                break

        except Exception as e:

            logger.exception(
                "❌ Could not get latest message "
                "from %s: %s",
                source,
                e,
            )

    # --------------------------------------------------------
    # Poll.
    # --------------------------------------------------------

    logger.info(
        "🚀 Starting polling loop..."
    )

    await poll_channels()


# ============================================================
# ENTRY POINT
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
