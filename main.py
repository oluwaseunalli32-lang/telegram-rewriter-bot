import os
import asyncio
import logging

from pathlib import Path

from dotenv import load_dotenv

from aiogram import Bot
from aiogram.types import (
    BufferedInputFile,
    InputMediaPhoto,
)

from telethon import (
    TelegramClient,
    errors,
)


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

env_path = (
    Path(__file__).parent
    / ".env"
)

load_dotenv(
    dotenv_path=env_path
)


# ============================================================
# IMPORTS
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
# CHANNELS
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
        "⚠️ Telegram disconnected. Reconnecting..."
    )

    try:

        await user_client.connect()

        if not user_client.is_connected():

            await user_client.start(
                phone=PHONE
            )

        return user_client.is_connected()

    except Exception:

        logger.exception(
            "❌ Reconnection failed."
        )

        return False


# ============================================================
# MEDIA DETECTION
# ============================================================

def get_image_media(
    msg,
):

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

        supported = {
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

        for attribute in attributes:

            filename = (
                getattr(
                    attribute,
                    "file_name",
                    "",
                )
                or ""
            )

            if (
                Path(
                    filename
                ).suffix.lower()
                in supported
            ):

                return document

    return None


# ============================================================
# MEDIA INFO
# ============================================================

def get_media_filename(
    msg,
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

    if getattr(
        msg,
        "photo",
        None,
    ):

        return (
            f"{msg.id}.jpg"
        )

    return (
        f"{msg.id}.media"
    )


def get_media_mime_type(
    msg,
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
# DOWNLOAD + OPENAI EDIT
# ============================================================

async def clean_message_media(
    msg,
):

    media = get_image_media(
        msg
    )

    if not media:
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

    filename = get_media_filename(
        msg
    )

    mime_type = get_media_mime_type(
        msg
    )

    logger.info(
        "🤖 [%s] Sending media to OpenAI image editor...",
        msg.id,
    )

    cleaned = (
        await remove_watermarks_from_bytes(
            image_bytes=original_bytes,
            filename=filename,
            mime_type=mime_type,
        )
    )

    if not cleaned:

        logger.error(
            "❌ [%s] OpenAI image editing failed.",
            msg.id,
        )

        return None

    logger.info(
        "✅ [%s] Clean still image ready.",
        msg.id,
    )

    return cleaned


# ============================================================
# TEXT
# ============================================================

def get_message_text(
    msg,
):

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
# SINGLE MESSAGE
# ============================================================

async def process_single_message(
    msg,
    target_id,
):

    original_text = get_message_text(
        msg
    )

    # --------------------------------------------------------
    # CAPTION REMAINS YOUR EXISTING LOGIC.
    # --------------------------------------------------------

    caption = await rewrite_text(
        original_text
    )

    media = get_image_media(
        msg
    )

    # --------------------------------------------------------
    # TEXT ONLY
    # --------------------------------------------------------

    if not media:

        if caption:

            await bot.send_message(
                chat_id=target_id,
                text=caption,
            )

            logger.info(
                "✅ [%s] Text posted.",
                msg.id,
            )

        return True

    # --------------------------------------------------------
    # MEDIA → OPENAI → STILL IMAGE
    # --------------------------------------------------------

    cleaned = (
        await clean_message_media(
            msg
        )
    )

    if not cleaned:
        return False

    await bot.send_photo(
        chat_id=target_id,
        photo=cleaned,
        caption=(
            caption[:1024]
            if caption
            else None
        ),
    )

    logger.info(
        "✅ [%s] Clean image posted.",
        msg.id,
    )

    return True


# ============================================================
# ALBUM
# ============================================================

async def process_album(
    messages,
    target_id,
):

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    logger.info(
        "🖼️ TELEGRAM ALBUM: %d items",
        len(messages),
    )

    # --------------------------------------------------------
    # Album caption.
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
    # Process EVERY image in album.
    #
    # We don't send anything until processing has completed.
    # --------------------------------------------------------

    processed = []

    for msg in messages:

        media = get_image_media(
            msg
        )

        if not media:
            continue

        cleaned = (
            await clean_message_media(
                msg
            )
        )

        if not cleaned:

            logger.error(
                "❌ [%s] Album item failed.",
                msg.id,
            )

            return False

        processed.append(
            cleaned
        )

    if not processed:

        return False

    # --------------------------------------------------------
    # Everything returned by the new pipeline is a still PNG.
    # Therefore the whole album can be sent as one photo album.
    # --------------------------------------------------------

    media_group = []

    for index, cleaned in enumerate(
        processed
    ):

        caption = None

        if (
            index == 0
            and album_caption
        ):

            caption = (
                album_caption[:1024]
            )

        media_group.append(
            InputMediaPhoto(
                media=cleaned,
                caption=caption,
            )
        )

    # Telegram limit: max 10 items per media group.
    for start in range(
        0,
        len(media_group),
        10,
    ):

        chunk = media_group[
            start:start + 10
        ]

        logger.info(
            "📤 Sending cleaned album chunk: %d item(s)",
            len(chunk),
        )

        await bot.send_media_group(
            chat_id=target_id,
            media=chunk,
        )

    logger.info(
        "✅ Album processed and posted."
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

        channel = await (
            user_client.get_entity(
                source_id
            )
        )

        last_id = (
            last_processed.get(
                source_id,
                0,
            )
        )

        messages = []

        async for msg in user_client.iter_messages(
            channel,
            min_id=last_id,
            reverse=True,
        ):

            messages.append(
                msg
            )

        if not messages:
            return

        # ----------------------------------------------------
        # Group Telegram albums.
        # ----------------------------------------------------

        groups = {}

        for msg in messages:

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

        ordered_groups = sorted(
            groups.values(),
            key=lambda group: min(
                x.id
                for x in group
            ),
        )

        # ----------------------------------------------------
        # Sequential processing.
        # ----------------------------------------------------

        for group in ordered_groups:

            first = group[0]

            grouped_id = getattr(
                first,
                "grouped_id",
                None,
            )

            # ================================================
            # ALBUM
            # ================================================

            if grouped_id:

                ids = [
                    msg.id
                    for msg in group
                ]

                logger.info(
                    "📚 Album %s detected with %d item(s).",
                    grouped_id,
                    len(group),
                )

                async with processing_lock:

                    if any(
                        msg_id
                        in processing_ids
                        for msg_id in ids
                    ):

                        continue

                    processing_ids.update(
                        ids
                    )

                try:

                    success = (
                        await process_album(
                            group,
                            target_id,
                        )
                    )

                    if success:

                        last_processed[
                            source_id
                        ] = max(
                            last_processed.get(
                                source_id,
                                0,
                            ),
                            max(ids),
                        )

                except Exception:

                    logger.exception(
                        "❌ Album processing failed."
                    )

                finally:

                    async with processing_lock:

                        for msg_id in ids:

                            processing_ids.discard(
                                msg_id
                            )

                continue

            # ================================================
            # SINGLE MESSAGE
            # ================================================

            msg = first

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
                    "📩 Processing message %s",
                    msg.id,
                )

                success = (
                    await process_single_message(
                        msg,
                        target_id,
                    )
                )

                if success:

                    last_processed[
                        source_id
                    ] = max(
                        last_processed.get(
                            source_id,
                            0,
                        ),
                        msg.id,
                    )

                else:

                    logger.error(
                        "❌ [%s] Processing failed. "
                        "Message will be retried.",
                        msg.id,
                    )

            except Exception:

                logger.exception(
                    "❌ [%s] Processing failed.",
                    msg.id,
                )

            finally:

                async with processing_lock:

                    processing_ids.discard(
                        msg.id
                    )

            await asyncio.sleep(
                2
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
        "🚀 Starting Telegram OpenAI Watermark Editor..."
    )

    logger.info(
        "   Caption: existing exact replacement"
    )

    logger.info(
        "   Images: OpenAI image edit"
    )

    logger.info(
        "   GIFs: representative frame → OpenAI edit → still image"
    )

    logger.info(
        "   Replicate: DISABLED"
    )

    logger.info(
        "   Telegram albums: ENABLED"
    )

    await user_client.start(
        phone=PHONE,
        force_sms=True,
    )

    if not user_client.is_connected():

        logger.error(
            "❌ Telegram client failed to connect."
        )

        return

    logger.info(
        "✅ User client connected!"
    )

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
            "✅ Client already registered: %s → %s",
            SOURCE_CHANNEL_ID,
            existing,
        )

    # --------------------------------------------------------
    # Don't process historical posts after deployment.
    # --------------------------------------------------------

    for client in (
        database.get_all_clients()
    ):

        source = client[
            "source"
        ]

        try:

            channel = await (
                user_client.get_entity(
                    source
                )
            )

            async for msg in user_client.iter_messages(
                channel,
                limit=1,
            ):

                last_processed[
                    source
                ] = msg.id

                logger.info(
                    "📌 Last message in %s: %s",
                    source,
                    msg.id,
                )

                break

        except Exception:

            logger.exception(
                "❌ Could not fetch latest message."
            )

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
