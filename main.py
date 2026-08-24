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
# ENVIRONMENT
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
# ENVIRONMENT VARIABLES
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
# PROCESSING
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

        mime = (
            getattr(
                document,
                "mime_type",
                "",
            )
            or ""
        ).lower()

        if mime.startswith(
            "image/"
        ):

            return document

        if mime.startswith(
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
                Path(filename)
                .suffix
                .lower()
                in supported
            ):

                return document

    return None


# ============================================================
# MEDIA FILENAME
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

        return f"{msg.id}.jpg"

    return f"{msg.id}.media"


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
# CLEAN ONE MEDIA
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

    original = await user_client.download_media(
        media,
        bytes,
    )

    if not original:

        logger.error(
            "❌ [%s] Download failed.",
            msg.id,
        )

        return None

    logger.info(
        "📦 [%s] Downloaded %d bytes.",
        msg.id,
        len(original),
    )

    filename = get_media_filename(
        msg
    )

    mime_type = get_media_mime_type(
        msg
    )

    cleaned = (
        await remove_watermarks_from_bytes(
            image_bytes=original,
            filename=filename,
            mime_type=mime_type,
        )
    )

    if not cleaned:

        logger.error(
            "❌ [%s] Watermark processing failed.",
            msg.id,
        )

        return None

    return cleaned


# ============================================================
# SINGLE MESSAGE
# ============================================================

async def process_single_message(
    msg,
    target_id,
):

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

    # IMPORTANT:
    # Existing caption processing remains unchanged.
    caption = await rewrite_text(
        original_text
    )

    media = get_image_media(
        msg
    )

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

    cleaned = (
        await clean_message_media(
            msg
        )
    )

    if not cleaned:

        return False

    # --------------------------------------------------------
    # Determine output type from the BufferedInputFile
    # filename.
    # --------------------------------------------------------

    filename = getattr(
        cleaned,
        "filename",
        "",
    ).lower()

    final_caption = (
        caption[:1024]
        if caption
        else None
    )

    # GIF
    if filename.endswith(
        ".gif"
    ):

        await bot.send_animation(
            chat_id=target_id,
            animation=cleaned,
            caption=final_caption,
        )

    # MP4
    elif filename.endswith(
        ".mp4"
    ):

        await bot.send_animation(
            chat_id=target_id,
            animation=cleaned,
            caption=final_caption,
        )

    # Normal image
    else:

        await bot.send_photo(
            chat_id=target_id,
            photo=cleaned,
            caption=final_caption,
        )

    logger.info(
        "✅ [%s] Cleaned media posted.",
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
    # Caption from first album item containing text.
    # --------------------------------------------------------

    album_caption = ""

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
            or getattr(
                msg,
                "caption",
                None,
            )
            or ""
        )

        if text:

            album_caption = (
                await rewrite_text(
                    text
                )
            )

            break

    # --------------------------------------------------------
    # Process every item before sending.
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
            (
                msg,
                cleaned,
            )
        )

    if not processed:
        return False

    # --------------------------------------------------------
    # Split normal photos from animated media.
    # --------------------------------------------------------

    photo_items = []

    animated_items = []

    for item in processed:

        msg, cleaned = item

        filename = getattr(
            cleaned,
            "filename",
            "",
        ).lower()

        if (
            filename.endswith(".gif")
            or filename.endswith(".mp4")
        ):

            animated_items.append(
                item
            )

        else:

            photo_items.append(
                item
            )

    # --------------------------------------------------------
    # Send photo album.
    # --------------------------------------------------------

    caption_used = False

    for start in range(
        0,
        len(photo_items),
        10,
    ):

        chunk = photo_items[
            start:start + 10
        ]

        media_group = []

        for msg, cleaned in chunk:

            caption = None

            if (
                not caption_used
                and album_caption
            ):

                caption = (
                    album_caption[:1024]
                )

                caption_used = True

            media_group.append(
                InputMediaPhoto(
                    media=cleaned,
                    caption=caption,
                )
            )

        if media_group:

            await bot.send_media_group(
                chat_id=target_id,
                media=media_group,
            )

    # --------------------------------------------------------
    # Send animations.
    #
    # Telegram cannot put GIF animations inside the same
    # InputMediaPhoto group, so animated items are sent as
    # animations after the photo group.
    # --------------------------------------------------------

    for msg, cleaned in animated_items:

        caption = None

        if (
            not caption_used
            and album_caption
        ):

            caption = (
                album_caption[:1024]
            )

            caption_used = True

        await bot.send_animation(
            chat_id=target_id,
            animation=cleaned,
            caption=caption,
        )

    logger.info(
        "✅ Album processed successfully."
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
        # Group by Telegram grouped_id.
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
        # Process groups sequentially.
        # ----------------------------------------------------

        for group in ordered_groups:

            first = group[0]

            grouped_id = getattr(
                first,
                "grouped_id",
                None,
            )

            # =================================================
            # ALBUM
            # =================================================

            if grouped_id:

                logger.info(
                    "📚 Album %s detected with %d message(s).",
                    grouped_id,
                    len(group),
                )

                album_keys = [
                    msg.id
                    for msg in group
                ]

                async with processing_lock:

                    if any(
                        msg_id
                        in processing_ids
                        for msg_id in album_keys
                    ):

                        continue

                    processing_ids.update(
                        album_keys
                    )

                try:

                    success = await process_album(
                        group,
                        target_id,
                    )

                    if success:

                        last_processed[
                            source_id
                        ] = max(
                            last_processed.get(
                                source_id,
                                0,
                            ),
                            max(album_keys),
                        )

                finally:

                    async with processing_lock:

                        for msg_id in album_keys:

                            processing_ids.discard(
                                msg_id
                            )

                continue

            # =================================================
            # SINGLE
            # =================================================

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
            "❌ Channel processing failed."
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
        "🚀 Starting Telegram Watermark Removal Bot..."
    )

    logger.info(
        "   Caption handling: EXACT username replacement"
    )

    logger.info(
        "   Red/faded-red @cappersfree detection: ENABLED"
    )

    logger.info(
        "   CF logo template detection: ENABLED"
    )

    logger.info(
        "   GIF pulsing-mask detection: ENABLED"
    )

    logger.info(
        "   OpenAI regeneration: DISABLED"
    )

    logger.info(
        "   Telegram album processing: ENABLED"
    )

    # --------------------------------------------------------
    # Telegram login.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Register default channel.
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
            "✅ Client already registered: %s → %s",
            SOURCE_CHANNEL_ID,
            existing,
        )

    # --------------------------------------------------------
    # Don't process old messages at startup.
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
                    "📌 Last message in %s: %s",
                    source,
                    msg.id,
                )

                break

        except Exception:

            logger.exception(
                "❌ Failed reading last message."
            )

    # --------------------------------------------------------
    # Start.
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
