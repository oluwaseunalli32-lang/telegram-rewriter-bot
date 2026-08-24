import os
import asyncio
import logging

from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT
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
)

from telethon import (
    TelegramClient,
    errors,
)

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

API_ID = int(
    os.getenv(
        "API_ID",
        "0",
    )
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
# TELEGRAM
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
            "✅ Telegram client reconnected."
        )

        return True

    except Exception as e:

        logger.error(
            "❌ Reconnection failed: %s",
            e,
        )

        return False


# ============================================================
# MEDIA DETECTION
# ============================================================

def get_image_media(msg):

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
# MEDIA INFORMATION
# ============================================================

def get_media_filename(
    msg,
    media,
):

    document = getattr(
        msg,
        "document",
        None,
    )

    if document:

        attributes = (
            getattr(
                document,
                "attributes",
                None,
            )
            or []
        )

        for attribute in attributes:

            filename = getattr(
                attribute,
                "file_name",
                None,
            )

            if filename:
                return filename

    return f"media_{msg.id}"


def get_media_mime(
    msg,
    media,
):

    document = getattr(
        msg,
        "document",
        None,
    )

    if document:

        return (
            getattr(
                document,
                "mime_type",
                "",
            )
            or ""
        )

    return "image/jpeg"


# ============================================================
# DOWNLOAD + REMOVE WATERMARK
# ============================================================

async def process_media_message(
    msg,
):

    media = get_image_media(
        msg
    )

    if not media:
        return None

    logger.info(
        "🖼️ [%s] Downloading media...",
        msg.id,
    )

    image_bytes = await user_client.download_media(
        media,
        bytes,
    )

    if not image_bytes:

        logger.error(
            "❌ [%s] Media download failed.",
            msg.id,
        )

        return None

    logger.info(
        "📦 [%s] Downloaded %d bytes.",
        msg.id,
        len(image_bytes),
    )

    filename = get_media_filename(
        msg,
        media,
    )

    mime_type = get_media_mime(
        msg,
        media,
    )

    logger.info(
        "🧹 [%s] Removing CF/@cappersfree watermark...",
        msg.id,
    )

    processed = await remove_watermarks_from_bytes(
        image_bytes=image_bytes,
        filename=filename,
        mime_type=mime_type,
    )

    if not processed:

        logger.error(
            "❌ [%s] Watermark removal failed.",
            msg.id,
        )

        return None

    logger.info(
        "✅ [%s] Watermark removal complete.",
        msg.id,
    )

    return processed


# ============================================================
# SINGLE IMAGE
# ============================================================

async def send_single_media(
    msg,
    target_id,
    caption,
):

    processed = await process_media_message(
        msg
    )

    if not processed:
        return False

    try:

        filename = get_media_filename(
            msg,
            get_image_media(msg),
        ).lower()

        # ----------------------------------------------------
        # Animated MP4
        # ----------------------------------------------------

        if (
            filename.endswith(".mp4")
            or filename.endswith(".mov")
            or filename.endswith(".webm")
        ):

            await bot.send_animation(
                chat_id=target_id,
                animation=processed,
                caption=(
                    caption[:1024]
                    if caption
                    else None
                ),
            )

        # ----------------------------------------------------
        # GIF
        # ----------------------------------------------------

        elif filename.endswith(
            ".gif"
        ):

            await bot.send_animation(
                chat_id=target_id,
                animation=processed,
                caption=(
                    caption[:1024]
                    if caption
                    else None
                ),
            )

        # ----------------------------------------------------
        # Normal image
        # ----------------------------------------------------

        else:

            await bot.send_photo(
                chat_id=target_id,
                photo=processed,
                caption=(
                    caption[:1024]
                    if caption
                    else None
                ),
            )

        logger.info(
            "✅ [%s] Processed media posted.",
            msg.id,
        )

        return True

    except Exception:

        logger.exception(
            "❌ [%s] Failed to send processed media.",
            msg.id,
        )

        return False


# ============================================================
# TELEGRAM ALBUM
# ============================================================

async def process_album(
    messages,
    target_id,
):

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    logger.info(
        "🖼️ ALBUM DETECTED: %d media items",
        len(messages),
    )

    logger.info(
        "🧹 Processing every album item before sending..."
    )

    processed_items = []

    for msg in messages:

        media = get_image_media(
            msg
        )

        if not media:
            continue

        processed = await process_media_message(
            msg
        )

        if processed:

            processed_items.append(
                (
                    msg,
                    processed,
                )
            )

        else:

            logger.error(
                "❌ [%s] Album item failed.",
                msg.id,
            )

    if not processed_items:

        logger.error(
            "❌ No album items were successfully processed."
        )

        return False

    # --------------------------------------------------------
    # Caption comes from the first message containing text.
    # --------------------------------------------------------

    album_caption = ""

    for msg, _ in processed_items:

        original_text = (
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

        if original_text:

            album_caption = await rewrite_text(
                original_text
            )

            break

    # --------------------------------------------------------
    # Telegram media groups support photos/videos.
    #
    # If everything is a normal image, send as one album.
    # --------------------------------------------------------

    media_group = []

    for index, (
        msg,
        processed,
    ) in enumerate(
        processed_items
    ):

        original_name = get_media_filename(
            msg,
            get_image_media(msg),
        ).lower()

        # Animated media cannot safely be mixed into a
        # normal photo album, so send it separately.
        if (
            original_name.endswith(".gif")
            or original_name.endswith(".mp4")
            or original_name.endswith(".mov")
            or original_name.endswith(".webm")
        ):

            logger.info(
                "🎞️ [%s] Animated album item will be sent separately.",
                msg.id,
            )

            try:

                await bot.send_animation(
                    chat_id=target_id,
                    animation=processed,
                    caption=(
                        album_caption[:1024]
                        if (
                            album_caption
                            and index == 0
                        )
                        else None
                    ),
                )

            except Exception:

                logger.exception(
                    "❌ [%s] Failed sending animated album item.",
                    msg.id,
                )

            continue

        media_group.append(
            InputMediaPhoto(
                media=processed,
                caption=(
                    album_caption[:1024]
                    if (
                        album_caption
                        and len(media_group) == 0
                    )
                    else None
                ),
            )
        )

    # --------------------------------------------------------
    # Send normal images together.
    # --------------------------------------------------------

    if media_group:

        # Telegram allows a maximum of 10 items in a media group.
        for start in range(
            0,
            len(media_group),
            10,
        ):

            chunk = media_group[
                start:start + 10
            ]

            await bot.send_media_group(
                chat_id=target_id,
                media=chunk,
            )

    logger.info(
        "✅ Album processed and posted."
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return True


# ============================================================
# CHANNEL PROCESSING
# ============================================================

async def process_channel(
    source_id,
    target_id,
):

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

        # ====================================================
        # GROUP BY TELEGRAM ALBUM
        # ====================================================

        index = 0

        while index < len(
            new_messages
        ):

            msg = new_messages[
                index
            ]

            grouped_id = getattr(
                msg,
                "grouped_id",
                None,
            )

            # ------------------------------------------------
            # Album
            # ------------------------------------------------

            if grouped_id:

                album = []

                while index < len(
                    new_messages
                ):

                    candidate = new_messages[
                        index
                    ]

                    if getattr(
                        candidate,
                        "grouped_id",
                        None,
                    ) != grouped_id:

                        break

                    album.append(
                        candidate
                    )

                    index += 1

                logger.info(
                    "📚 Telegram album grouped_id=%s contains %d messages.",
                    grouped_id,
                    len(album),
                )

                # Mark all album messages as being processed.
                async with processing_lock:

                    for album_msg in album:

                        if album_msg.id in processing_ids:

                            logger.warning(
                                "⚠️ [%s] Already processing.",
                                album_msg.id,
                            )

                        processing_ids.add(
                            album_msg.id
                        )

                try:

                    success = await process_album(
                        album,
                        target_id,
                    )

                    if success:

                        for album_msg in album:

                            last_processed[source_id] = max(
                                last_processed.get(
                                    source_id,
                                    0,
                                ),
                                album_msg.id,
                            )

                    await asyncio.sleep(
                        2
                    )

                finally:

                    async with processing_lock:

                        for album_msg in album:

                            processing_ids.discard(
                                album_msg.id
                            )

                continue

            # ------------------------------------------------
            # Normal message
            # ------------------------------------------------

            index += 1

            async with processing_lock:

                if msg.id in processing_ids:

                    continue

                processing_ids.add(
                    msg.id
                )

            try:

                logger.info(
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )

                logger.info(
                    "📩 Processing message ID: %s",
                    msg.id,
                )

                original_text = (
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

                # =================================================
                # CAPTION — UNTOUCHED EXCEPT EXISTING REPLACEMENT
                # =================================================

                rewritten_text = await rewrite_text(
                    original_text
                )

                media = get_image_media(
                    msg
                )

                if media:

                    logger.info(
                        "🖼️ [%s] IMAGE/GIF DETECTED",
                        msg.id,
                    )

                    success = await send_single_media(
                        msg=msg,
                        target_id=target_id,
                        caption=rewritten_text,
                    )

                    if not success:

                        logger.error(
                            "❌ [%s] Media was NOT posted.",
                            msg.id,
                        )

                else:

                    if rewritten_text:

                        await bot.send_message(
                            chat_id=target_id,
                            text=rewritten_text,
                        )

                        logger.info(
                            "📝 [%s] Text message posted.",
                            msg.id,
                        )

                last_processed[source_id] = max(
                    last_processed.get(
                        source_id,
                        0,
                    ),
                    msg.id,
                )

                await asyncio.sleep(
                    2
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

    except errors.rpcerrorlist.AuthKeyError:

        logger.exception(
            "❌ Telegram authentication error."
        )

    except Exception:

        logger.exception(
            "❌ Channel processing failed for %s.",
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

        clients = database.get_all_clients()

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
        "🚀 Starting Telegram Watermark Removal Bot..."
    )

    logger.info(
        "   Caption handling: EXACT username replacement"
    )

    logger.info(
        "   Image handling: DIRECT WATERMARK REMOVAL"
    )

    logger.info(
        "   Red watermark: HSV + connected components"
    )

    logger.info(
        "   CF graphic: temporal/pulsing mask for GIFs"
    )

    logger.info(
        "   Albums: ENABLED"
    )

    await user_client.start(
        phone=PHONE,
        force_sms=True,
    )

    if not user_client.is_connected():

        await user_client.connect()

    if not user_client.is_connected():

        logger.error(
            "❌ Failed to connect Telegram client."
        )

        return

    logger.info(
        "✅ User client connected!"
    )

    # ========================================================
    # DATABASE CHANNEL
    # ========================================================

    existing = database.get_target_for_source(
        SOURCE_CHANNEL_ID
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

    # ========================================================
    # START FROM NEWEST MESSAGE
    # ========================================================

    for client in database.get_all_clients():

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

                last_processed[source] = (
                    msg.id
                )

                logger.info(
                    "📌 Last message in %s: %s",
                    source,
                    msg.id,
                )

        except Exception:

            logger.exception(
                "❌ Could not fetch latest message from %s.",
                source,
            )

    # ========================================================
    # POLLING
    # ========================================================

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
