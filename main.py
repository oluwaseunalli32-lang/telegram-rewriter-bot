import os
import asyncio
import logging
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv


# ============================================================
# LOAD .ENV BEFORE AI PROCESSOR IMPORT
# ============================================================

env_path = Path(__file__).parent / ".env"

load_dotenv(
    dotenv_path=env_path
)


# ============================================================
# IMPORTS
# ============================================================

from aiogram import Bot
from aiogram.types import (
    BufferedInputFile,
    InputMediaPhoto,
    InputMediaVideo,
)

from telethon import TelegramClient, errors

import database

from ai_processor import (
    remove_watermark_from_bytes,
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
    "telegram_reposter"
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
# DEFAULT CHANNEL
# ============================================================

SOURCE_CHANNEL_ID = -1003593544389

TARGET_CHANNEL_ID = -1004415621706


# ============================================================
# PROCESSING STATE
# ============================================================

processing_ids = set()

processing_lock = asyncio.Lock()

last_processed = {}


# ============================================================
# ALBUM BUFFER
# ============================================================

# Telegram sends grouped media as separate messages.
#
# We temporarily collect messages by grouped_id so:
#
# image 1
# image 2
# image 3
#
# becomes one album when posted to the target.
#

album_buffers = defaultdict(list)

album_tasks = {}

ALBUM_WAIT_SECONDS = 3


# ============================================================
# CONNECTION
# ============================================================

async def ensure_connection():

    if user_client.is_connected():

        return True

    logger.warning(
        "⚠️ Telegram client disconnected. "
        "Reconnecting..."
    )

    try:

        await user_client.connect()

        if not user_client.is_connected():

            await user_client.start(
                phone=PHONE
            )

        logger.info(
            "✅ Telegram client reconnected."
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
    Return photo/image/video/GIF media.
    """

    if getattr(
        msg,
        "photo",
        None,
    ):

        return msg.photo

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
                Path(
                    filename
                )
                .suffix
                .lower()
            )

            if extension in {
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
                ".webm",
            }:

                return document

    media = getattr(
        msg,
        "media",
        None,
    )

    if media and hasattr(
        media,
        "photo",
    ):

        return media

    return None


# ============================================================
# DETERMINE MEDIA TYPE
# ============================================================

def media_is_video(
    msg,
):
    """
    Determine whether Telegram media is video/GIF/MP4.
    """

    document = getattr(
        msg,
        "document",
        None,
    )

    if not document:

        return False

    mime_type = (
        getattr(
            document,
            "mime_type",
            "",
        )
        or ""
    ).lower()

    if mime_type.startswith(
        "video/"
    ):

        return True

    attributes = (
        getattr(
            document,
            "attributes",
            None,
        )
        or []
    )

    for attribute in attributes:

        filename = (
            getattr(
                attribute,
                "file_name",
                "",
            )
            or ""
        ).lower()

        if filename.endswith(
            (
                ".gif",
                ".mp4",
                ".mov",
                ".webm",
            )
        ):

            return True

    return False


# ============================================================
# DOWNLOAD + CLEAN ONE MEDIA ITEM
# ============================================================

async def clean_message_media(
    msg,
):
    """
    Download and remove only the watermark.
    """

    media = get_image_media(
        msg
    )

    if not media:

        return None

    logger.info(
        "⬇️ [%s] Downloading media...",
        msg.id,
    )

    original = (
        await user_client.download_media(
            media,
            bytes,
        )
    )

    if not original:

        logger.error(
            "❌ [%s] Could not download media.",
            msg.id,
        )

        return None

    logger.info(
        "📦 [%s] Downloaded %d bytes.",
        msg.id,
        len(original),
    )

    logger.info(
        "🧹 [%s] Removing CF watermark...",
        msg.id,
    )

    cleaned = (
        await remove_watermark_from_bytes(
            original
        )
    )

    if not cleaned:

        logger.error(
            "❌ [%s] Watermark processing failed.",
            msg.id,
        )

        return None

    logger.info(
        "✅ [%s] Media cleaned: %d bytes.",
        msg.id,
        len(cleaned),
    )

    return {
        "message": msg,
        "bytes": cleaned,
        "is_video": media_is_video(msg),
    }


# ============================================================
# SEND SINGLE MEDIA
# ============================================================

async def send_single_media(
    item,
    caption,
    target_id,
):
    """
    Send one cleaned image/video.

    Caption is passed exactly as received.
    """

    filename = (
        "cleaned.mp4"
        if item["is_video"]
        else "cleaned.png"
    )

    file = BufferedInputFile(
        item["bytes"],
        filename=filename,
    )

    if item["is_video"]:

        await bot.send_video(
            chat_id=target_id,
            video=file,
            caption=caption or None,
        )

    else:

        await bot.send_photo(
            chat_id=target_id,
            photo=file,
            caption=caption or None,
        )


# ============================================================
# SEND ALBUM
# ============================================================

async def send_album(
    items,
    caption,
    target_id,
):
    """
    Send all cleaned album items together.

    Telegram media groups support 2-10 items.
    """

    if not items:

        return False

    # --------------------------------------------------------
    # One item isn't an album.
    # --------------------------------------------------------

    if len(items) == 1:

        await send_single_media(
            items[0],
            caption,
            target_id,
        )

        return True

    # --------------------------------------------------------
    # Telegram media group maximum is 10.
    #
    # If an exceptionally large Telegram album is received,
    # split it into consecutive groups.
    # --------------------------------------------------------

    for start in range(
        0,
        len(items),
        10,
    ):

        chunk = items[
            start:start + 10
        ]

        media = []

        for index, item in enumerate(
            chunk
        ):

            filename = (
                f"cleaned_{start + index}.mp4"
                if item["is_video"]
                else f"cleaned_{start + index}.png"
            )

            file = BufferedInputFile(
                item["bytes"],
                filename=filename,
            )

            # Caption ONLY goes on first item.
            item_caption = (
                caption
                if start == 0
                and index == 0
                else None
            )

            if item["is_video"]:

                media.append(
                    InputMediaVideo(
                        media=file,
                        caption=item_caption,
                        supports_streaming=True,
                    )
                )

            else:

                media.append(
                    InputMediaPhoto(
                        media=file,
                        caption=item_caption,
                    )
                )

        logger.info(
            "📤 Sending album chunk: %d item(s)",
            len(media),
        )

        await bot.send_media_group(
            chat_id=target_id,
            media=media,
        )

    return True


# ============================================================
# PROCESS ALBUM
# ============================================================

async def process_album(
    grouped_id,
    target_id,
):
    """
    Process every message in a Telegram grouped album.
    """

    messages = album_buffers.pop(
        grouped_id,
        [],
    )

    album_tasks.pop(
        grouped_id,
        None,
    )

    if not messages:

        return

    messages.sort(
        key=lambda msg: msg.id
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    logger.info(
        "🖼️ ALBUM DETECTED"
    )

    logger.info(
        "🖼️ Group ID: %s",
        grouped_id,
    )

    logger.info(
        "🖼️ Items: %d",
        len(messages),
    )

    # --------------------------------------------------------
    # Caption comes from the first message that has one.
    #
    # We DO NOT modify it.
    # --------------------------------------------------------

    caption = ""

    for msg in messages:

        text = (
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
            or ""
        )

        if text:

            caption = text
            break

    # --------------------------------------------------------
    # Clean every image.
    # --------------------------------------------------------

    cleaned_items = []

    for msg in messages:

        item = (
            await clean_message_media(
                msg
            )
        )

        if item:

            cleaned_items.append(
                item
            )

        else:

            logger.error(
                "❌ [%s] Album item failed.",
                msg.id,
            )

    # --------------------------------------------------------
    # Never send the original media if cleaning failed.
    # --------------------------------------------------------

    if len(cleaned_items) != len(
        messages
    ):

        logger.error(
            "❌ Album %s was NOT posted because "
            "one or more items failed.",
            grouped_id,
        )

        return

    # --------------------------------------------------------
    # Send everything together.
    # --------------------------------------------------------

    try:

        await send_album(
            cleaned_items,
            caption,
            target_id,
        )

        logger.info(
            "✅ Album %s posted successfully.",
            grouped_id,
        )

    except Exception:

        logger.exception(
            "❌ Failed sending album %s.",
            grouped_id,
        )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# ALBUM WAIT
# ============================================================

async def schedule_album(
    grouped_id,
    target_id,
):
    """
    Wait briefly for all messages belonging to the same
    Telegram album to arrive.
    """

    await asyncio.sleep(
        ALBUM_WAIT_SECONDS
    )

    await process_album(
        grouped_id,
        target_id,
    )


# ============================================================
# PROCESS CHANNEL
# ============================================================

async def process_channel(
    source_id,
    target_id,
):
    """
    Poll one source channel.

    Handles:
        - normal images
        - GIFs
        - videos
        - Telegram albums
        - text-only messages
    """

    if not await ensure_connection():

        return

    try:

        channel = await user_client.get_entity(
            source_id
        )

        last_id = last_processed.get(
            source_id,
            0,
        )

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
        # First collect grouped albums.
        # ----------------------------------------------------

        grouped_messages = defaultdict(
            list
        )

        normal_messages = []

        for msg in new_messages:

            if getattr(
                msg,
                "grouped_id",
                None,
            ):

                grouped_messages[
                    msg.grouped_id
                ].append(
                    msg
                )

            else:

                normal_messages.append(
                    msg
                )

        # ----------------------------------------------------
        # Start album processing.
        #
        # A short delay allows Telegram's album messages
        # to arrive before we process the group.
        # ----------------------------------------------------

        for grouped_id, messages in (
            grouped_messages.items()
        ):

            album_buffers[
                grouped_id
            ].extend(
                messages
            )

            if grouped_id not in album_tasks:

                album_tasks[
                    grouped_id
                ] = asyncio.create_task(
                    schedule_album(
                        grouped_id,
                        target_id,
                    )
                )

        # ----------------------------------------------------
        # Normal non-album messages.
        # ----------------------------------------------------

        for msg in normal_messages:

            async with processing_lock:

                if msg.id in processing_ids:

                    logger.warning(
                        "⚠️ [%s] Already processing.",
                        msg.id,
                    )

                    continue

                processing_ids.add(
                    msg.id
                )

            try:

                logger.info(
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )

                logger.info(
                    "📩 Processing message %s",
                    msg.id,
                )

                media = get_image_media(
                    msg
                )

                # ------------------------------------------------
                # IMAGE / GIF / VIDEO
                # ------------------------------------------------

                if media:

                    item = (
                        await clean_message_media(
                            msg
                        )
                    )

                    if not item:

                        logger.error(
                            "❌ [%s] Media was NOT posted "
                            "because cleaning failed.",
                            msg.id,
                        )

                    else:

                        caption = (
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
                            or ""
                        )

                        await send_single_media(
                            item,
                            caption,
                            target_id,
                        )

                        logger.info(
                            "✅ [%s] Cleaned media posted.",
                            msg.id,
                        )

                # ------------------------------------------------
                # TEXT ONLY
                # ------------------------------------------------

                else:

                    text = (
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
                        or ""
                    )

                    if text:

                        # IMPORTANT:
                        # Exact caption/text.
                        await bot.send_message(
                            chat_id=target_id,
                            text=text,
                        )

                        logger.info(
                            "✅ [%s] Text posted unchanged.",
                            msg.id,
                        )

            except Exception:

                logger.exception(
                    "❌ [%s] Message processing failed.",
                    msg.id,
                )

            finally:

                async with processing_lock:

                    processing_ids.discard(
                        msg.id
                    )

                last_processed[
                    source_id
                ] = max(
                    last_processed.get(
                        source_id,
                        0,
                    ),
                    msg.id,
                )

        # ----------------------------------------------------
        # Mark album messages as seen.
        # ----------------------------------------------------

        for messages in (
            grouped_messages.values()
        ):

            for msg in messages:

                last_processed[
                    source_id
                ] = max(
                    last_processed.get(
                        source_id,
                        0,
                    ),
                    msg.id,
                )

    except errors.rpcerrorlist.AuthKeyError as e:

        logger.error(
            "❌ Telegram authentication error: %s",
            e,
        )

        try:

            await user_client.start(
                phone=PHONE
            )

        except Exception:

            logger.exception(
                "❌ Could not restart Telegram client."
            )

    except Exception:

        logger.exception(
            "❌ Error processing channel %s.",
            source_id,
        )


# ============================================================
# POLLING
# ============================================================

async def poll_channels():

    while True:

        if not await ensure_connection():

            await asyncio.sleep(
                10
            )

            continue

        clients = (
            database.get_all_clients()
        )

        if not clients:

            await asyncio.sleep(
                10
            )

            continue

        for client in clients:

            await process_channel(
                client["source"],
                client["target"],
            )

        await asyncio.sleep(
            5
        )


# ============================================================
# MAIN
# ============================================================

async def main():

    logger.info(
        "🚀 Starting Telegram watermark-removal bot..."
    )

    logger.info(
        "🧹 OpenAI: DISABLED"
    )

    logger.info(
        "🧹 Image regeneration: DISABLED"
    )

    logger.info(
        "🧹 Caption modification: DISABLED"
    )

    logger.info(
        "🧹 CF watermark removal: ENABLED"
    )

    logger.info(
        "🖼️ Album processing: ENABLED"
    )

    # --------------------------------------------------------
    # Telegram login
    # --------------------------------------------------------

    await user_client.start(
        phone=PHONE,
        force_sms=True,
    )

    if not user_client.is_connected():

        await user_client.connect()

    if not user_client.is_connected():

        logger.error(
            "❌ Failed to connect."
        )

        return

    logger.info(
        "✅ Telegram user client connected."
    )

    # --------------------------------------------------------
    # Register default source → target.
    # --------------------------------------------------------

    existing = (
        database.get_target_for_source(
            SOURCE_CHANNEL_ID
        )
    )

    if existing is None:

        database.add_client(
            SOURCE_CHANNEL_ID,
            TARGET_CHANNEL_ID,
        )

        logger.info(
            "📝 Registered %s → %s",
            SOURCE_CHANNEL_ID,
            TARGET_CHANNEL_ID,
        )

    else:

        logger.info(
            "✅ Client already registered: "
            "%s → %s",
            SOURCE_CHANNEL_ID,
            existing,
        )

    # --------------------------------------------------------
    # Start from newest message.
    # --------------------------------------------------------

    for client in (
        database.get_all_clients()
    ):

        source = client[
            "source"
        ]

        try:

            channel = (
                await user_client.get_entity(
                    source
                )
            )

            async for msg in (
                user_client.iter_messages(
                    channel,
                    limit=1,
                )
            ):

                last_processed[
                    source
                ] = msg.id

                logger.info(
                    "📌 Starting after message %s "
                    "in %s",
                    msg.id,
                    source,
                )

        except Exception:

            logger.exception(
                "❌ Could not get latest message "
                "from %s",
                source,
            )

    # --------------------------------------------------------
    # Poll
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
