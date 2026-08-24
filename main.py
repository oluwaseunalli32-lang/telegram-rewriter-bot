import os
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
# IMPORTS THAT READ ENVIRONMENT
# ============================================================

import database

from ai_processor import (
    remove_watermarks_from_bytes,
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

# Telegram albums normally arrive very quickly as several
# messages sharing the same grouped_id.
#
# We wait briefly before collecting the group so we don't
# accidentally process only the first image.

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
# CAPTION PROCESSING
# ============================================================

def replace_username(
    text: str,
) -> str:

    """
    EXISTING CAPTION BEHAVIOR.

    Only:

        1. remove *
        2. replace @cappersfree

    No AI.
    No paraphrasing.
    No caption rewriting.
    """

    if not text:
        return text

    result = text

    result = result.replace(
        "*",
        "",
    )

    if NEW_MENTION:

        pattern = (
            __import__("re")
            .escape(
                OLD_MENTION
            )
        )

        result = __import__(
            "re"
        ).sub(
            pattern,
            NEW_MENTION,
            result,
            flags=__import__(
                "re"
            ).IGNORECASE,
        )

    else:

        logger.warning(
            "⚠️ NEW_MENTION is empty."
        )

    return result


async def rewrite_text(
    original_text: str,
) -> str:

    """
    Kept identical in behavior to your existing script.
    """

    result = replace_username(
        original_text
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    logger.info(
        "📝 CAPTION PROCESSING"
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
        "👤 OLD:      %r",
        OLD_MENTION,
    )

    logger.info(
        "👤 NEW:      %r",
        NEW_MENTION,
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    return result


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
    Return media that can be downloaded and cleaned.
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

            if extension in (
                supported_extensions
            ):

                return document

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
# DOWNLOAD + CLEAN ONE MEDIA
# ============================================================

async def clean_message_media(
    msg,
):
    """
    Download one Telegram media object and remove watermark(s).
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
        "🧹 [%s] Removing watermark(s)...",
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

    cleaned_bytes, media_type = (
        cleaned
    )

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
# PROCESS ONE IMAGE
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

    # --------------------------------------------------------
    # CAPTION PROCESSING
    # --------------------------------------------------------

    caption = await rewrite_text(
        original_text
    )

    media = get_image_media(
        msg
    )

    if media:

        cleaned = (
            await clean_message_media(
                msg
            )
        )

        if not cleaned:

            return False

        filename = (
            f"cleaned_{msg.id}"
        )

        if cleaned["type"] == "video":

            file = BufferedInputFile(
                cleaned["bytes"],
                filename
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

        else:

            file = BufferedInputFile(
                cleaned["bytes"],
                filename
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
    # TEXT ONLY
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
# COLLECT COMPLETE ALBUM
# ============================================================

async def collect_album(
    channel,
    grouped_id,
    minimum_id,
):
    """
    Wait briefly and then collect all recent messages that
    belong to the same Telegram album.
    """

    await asyncio.sleep(
        ALBUM_SETTLE_SECONDS
    )

    album_messages = {}

    # --------------------------------------------------------
    # First use messages already fetched.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Re-fetch recent messages.
    #
    # This catches album items that arrived a fraction later.
    # --------------------------------------------------------

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
    Process every image in an album FIRST.

    Nothing is sent until every item has been successfully
    cleaned.

    This prevents a half-processed album from being posted.
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
    # Caption.
    #
    # Telegram normally puts the album caption on the first
    # message. We preserve your existing processing exactly.
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
    # Clean every item.
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
    # Telegram allows max 10 media per media group.
    #
    # If an unusually large group arrives, send chunks.
    # --------------------------------------------------------

    chunks = [
        cleaned_items[
            i:i + TELEGRAM_ALBUM_SIZE
        ]
        for i in range(
            0,
            len(cleaned_items),
            TELEGRAM_ALBUM_SIZE,
        )
    ]

    caption_used = False

    for chunk_index, chunk in enumerate(
        chunks
    ):

        media_group = []

        for item_index, item in enumerate(
            chunk
        ):

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
        # Group messages by Telegram grouped_id.
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
        # Process groups in chronological order.
        # ----------------------------------------------------

        ordered_groups = sorted(
            groups.values(),
            key=lambda group: min(
                msg.id
                for msg in group
            ),
        )

        for initial_group in ordered_groups:

            first_message = (
                initial_group[0]
            )

            grouped_id = getattr(
                first_message,
                "grouped_id",
                None,
            )

            if grouped_id:

                # --------------------------------------------
                # Collect the complete album.
                # --------------------------------------------

                messages = (
                    await collect_album(
                        channel,
                        grouped_id,
                        last_id,
                    )
                )

                # If re-fetch didn't find anything, use what
                # was already discovered.
                if not messages:

                    messages = sorted(
                        initial_group,
                        key=lambda m: m.id,
                    )

                message_ids = [
                    (
                        source_id,
                        msg.id,
                    )
                    for msg in messages
                ]

                async with processing_lock:

                    if any(
                        item in processing_ids
                        for item in message_ids
                    ):

                        logger.warning(
                            "⚠️ Album already processing. "
                            "Skipping duplicate."
                        )

                        continue

                    processing_ids.update(
                        message_ids
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

                        for item in message_ids:

                            processing_ids.discard(
                                item
                            )

            else:

                # --------------------------------------------
                # Normal single message.
                # --------------------------------------------

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

            # Small delay between groups.
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
        "   Image: watermark detection + OpenCV inpainting"
    )

    logger.info(
        "   AI image generation: DISABLED"
    )

    logger.info(
        "   Telegram albums: ENABLED"
    )

    # --------------------------------------------------------
    # Start Telegram user client.
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
