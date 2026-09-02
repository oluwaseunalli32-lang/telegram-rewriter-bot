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
# LOAD ENVIRONMENT
# ============================================================

env_path = Path(__file__).parent / ".env"

load_dotenv(
    dotenv_path=env_path
)


# ============================================================
# DATABASE
# ============================================================

import database


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
# ENVIRONMENT VARIABLES
# ============================================================

BOT_TOKEN = os.getenv(
    "BOT_TOKEN",
    "",
).strip()

API_ID = int(
    os.getenv(
        "API_ID",
        "0",
    )
)

API_HASH = os.getenv(
    "API_HASH",
    "",
).strip()

PHONE = os.getenv(
    "PHONE_NUMBER",
    "",
).strip()


if (
    not BOT_TOKEN
    or not API_ID
    or not API_HASH
    or not PHONE
):

    logger.error(
        "❌ Missing required environment variables."
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
# DEFAULT SOURCE/TARGET
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

        if user_client.is_connected():

            logger.info(
                "✅ Telegram client reconnected."
            )

            return True

    except Exception:

        logger.exception(
            "❌ Telegram reconnection failed."
        )

    return False


# ============================================================
# CAPTION PROCESSING
# ============================================================

async def process_caption(
    original_text: str,
) -> str:

    """
    Caption behavior ONLY:

    1. Remove *
    2. Replace @cappersfree with NEW_MENTION

    No AI.
    No paraphrasing.
    No rewriting.
    """

    old_mention = os.getenv(
        "OLD_MENTION",
        "@cappersfree",
    ).strip()

    new_mention = os.getenv(
        "NEW_MENTION",
        "",
    ).strip()

    if (
        new_mention
        and not new_mention.startswith("@")
    ):
        new_mention = "@" + new_mention

    if not original_text:

        return ""

    result = original_text.replace(
        "*",
        "",
    )

    if new_mention:

        import re

        result = re.sub(
            re.escape(old_mention),
            new_mention,
            result,
            flags=re.IGNORECASE,
        )

    return result


# ============================================================
# MESSAGE TEXT
# ============================================================

def get_message_text(
    msg,
) -> str:

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
# MEDIA DETECTION
# ============================================================

def get_media(
    msg,
):

    # --------------------------------------------------------
    # Telegram photo.
    # --------------------------------------------------------

    if getattr(
        msg,
        "photo",
        None,
    ):

        return msg.photo

    # --------------------------------------------------------
    # Telegram document.
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

        if (
            mime_type.startswith(
                "image/"
            )
            or mime_type.startswith(
                "video/"
            )
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
                Path(
                    filename
                )
                .suffix
                .lower()
            )

            if extension in supported_extensions:

                return document

    return None


# ============================================================
# MEDIA TYPE
# ============================================================

def get_media_type(
    msg,
):

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
            "video/"
        ):

            return "video"

        if mime_type in {
            "image/gif",
        }:

            return "animation"

    if getattr(
        msg,
        "photo",
        None,
    ):

        return "photo"

    return "document"


# ============================================================
# ORIGINAL MEDIA FILENAME
# ============================================================

def get_filename(
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


# ============================================================
# DOWNLOAD ORIGINAL MEDIA
# ============================================================

async def download_original_media(
    msg,
):

    media = get_media(
        msg
    )

    if not media:

        logger.error(
            "❌ [%s] No supported media found.",
            msg.id,
        )

        return None

    logger.info(
        "⬇️ [%s] Downloading ORIGINAL media...",
        msg.id,
    )

    try:

        media_bytes = (
            await user_client.download_media(
                media,
                bytes,
            )
        )

    except Exception:

        logger.exception(
            "❌ [%s] Media download failed.",
            msg.id,
        )

        return None

    if not media_bytes:

        logger.error(
            "❌ [%s] Download returned no data.",
            msg.id,
        )

        return None

    logger.info(
        "📦 [%s] Original media: %d bytes",
        msg.id,
        len(media_bytes),
    )

    return media_bytes


# ============================================================
# PREPARE AIROGRAM FILE
# ============================================================

def make_input_file(
    msg,
    media_bytes: bytes,
):

    media_type = get_media_type(
        msg
    )

    filename = get_filename(
        msg
    )

    # --------------------------------------------------------
    # Preserve GIF.
    # --------------------------------------------------------

    if media_type == "animation":

        if not filename.lower().endswith(
            ".gif"
        ):

            filename = (
                "original.gif"
            )

    # --------------------------------------------------------
    # Preserve MP4/video.
    # --------------------------------------------------------

    elif media_type == "video":

        if not any(
            filename.lower().endswith(
                extension
            )
            for extension in (
                ".mp4",
                ".mov",
                ".m4v",
                ".webm",
            )
        ):

            filename = (
                "original.mp4"
            )

    # --------------------------------------------------------
    # Normal photo.
    # --------------------------------------------------------

    elif media_type == "photo":

        filename = (
            "original.jpg"
        )

    return BufferedInputFile(
        media_bytes,
        filename=filename,
    )


# ============================================================
# SEND ORIGINAL MEDIA
# ============================================================

async def send_original_media(
    msg,
    target_id,
    caption,
):

    media = get_media(
        msg
    )

    if not media:

        return False

    media_bytes = (
        await download_original_media(
            msg
        )
    )

    if not media_bytes:

        return False

    media_type = get_media_type(
        msg
    )

    input_file = make_input_file(
        msg,
        media_bytes,
    )

    final_caption = (
        caption[:1024]
        if caption
        else None
    )

    try:

        # ====================================================
        # PHOTO
        # ====================================================

        if media_type == "photo":

            await bot.send_photo(
                chat_id=target_id,
                photo=input_file,
                caption=final_caption,
            )

        # ====================================================
        # GIF / ANIMATION
        # ====================================================

        elif media_type == "animation":

            await bot.send_animation(
                chat_id=target_id,
                animation=input_file,
                caption=final_caption,
            )

        # ====================================================
        # VIDEO
        # ====================================================

        elif media_type == "video":

            await bot.send_video(
                chat_id=target_id,
                video=input_file,
                caption=final_caption,
            )

        # ====================================================
        # OTHER DOCUMENT
        # ====================================================

        else:

            await bot.send_document(
                chat_id=target_id,
                document=input_file,
                caption=final_caption,
            )

        logger.info(
            "✅ [%s] ORIGINAL media reposted.",
            msg.id,
        )

        return True

    except Exception:

        logger.exception(
            "❌ [%s] Failed to send original media.",
            msg.id,
        )

        return False


# ============================================================
# SINGLE MESSAGE
# ============================================================

async def process_single_message(
    msg,
    target_id,
):

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    logger.info(
        "📩 Processing message %s",
        msg.id,
    )

    original_text = (
        get_message_text(
            msg
        )
    )

    # --------------------------------------------------------
    # Caption ONLY.
    # --------------------------------------------------------

    caption = await process_caption(
        original_text
    )

    if original_text != caption:

        logger.info(
            "✏️ [%s] Caption updated.",
            msg.id,
        )

        logger.info(
            "📝 Original: %r",
            original_text,
        )

        logger.info(
            "📝 Final: %r",
            caption,
        )

    media = get_media(
        msg
    )

    # --------------------------------------------------------
    # TEXT ONLY.
    # --------------------------------------------------------

    if not media:

        if caption:

            await bot.send_message(
                chat_id=target_id,
                text=caption,
            )

            logger.info(
                "✅ [%s] Text message reposted.",
                msg.id,
            )

        return True

    # --------------------------------------------------------
    # ORIGINAL MEDIA.
    # --------------------------------------------------------

    logger.info(
        "🖼️ [%s] MEDIA DETECTED",
        msg.id,
    )

    logger.info(
        "⏸️ AI IMAGE GENERATION DISABLED"
    )

    logger.info(
        "⏸️ WATERMARK REMOVAL DISABLED"
    )

    return await send_original_media(
        msg,
        target_id,
        caption,
    )


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
        "📚 TELEGRAM ALBUM: %d item(s)",
        len(messages),
    )

    # --------------------------------------------------------
    # Find album caption.
    # --------------------------------------------------------

    album_caption = ""

    for msg in messages:

        text = (
            get_message_text(
                msg
            )
        )

        if text:

            album_caption = (
                await process_caption(
                    text
                )
            )

            break

    # --------------------------------------------------------
    # Download all original media FIRST.
    # --------------------------------------------------------

    items = []

    for msg in messages:

        media = get_media(
            msg
        )

        if not media:

            continue

        media_bytes = (
            await download_original_media(
                msg
            )
        )

        if not media_bytes:

            logger.error(
                "❌ [%s] Album item could not be downloaded.",
                msg.id,
            )

            return False

        media_type = (
            get_media_type(
                msg
            )
        )

        # Photo album can contain actual photos.
        # GIF/video items are handled individually below.
        items.append(
            (
                msg,
                media_bytes,
                media_type,
            )
        )

    if not items:

        logger.error(
            "❌ Album contains no supported media."
        )

        return False

    # --------------------------------------------------------
    # Separate photos from GIF/video.
    # --------------------------------------------------------

    photos = []

    non_photos = []

    for item in items:

        if item[2] == "photo":

            photos.append(
                item
            )

        else:

            non_photos.append(
                item
            )

    # --------------------------------------------------------
    # Send photo album.
    # --------------------------------------------------------

    if photos:

        photo_media = []

        for index, (
            msg,
            media_bytes,
            _,
        ) in enumerate(
            photos
        ):

            filename = (
                "original.jpg"
            )

            photo_file = (
                BufferedInputFile(
                    media_bytes,
                    filename=filename,
                )
            )

            caption = None

            if (
                not photo_media
                and album_caption
            ):

                caption = (
                    album_caption[:1024]
                )

            photo_media.append(
                InputMediaPhoto(
                    media=photo_file,
                    caption=caption,
                )
            )

        # Telegram allows max 10 items per media group.
        for start in range(
            0,
            len(photo_media),
            10,
        ):

            chunk = photo_media[
                start:start + 10
            ]

            logger.info(
                "📤 Sending ORIGINAL photo album "
                "with %d item(s).",
                len(chunk),
            )

            await bot.send_media_group(
                chat_id=target_id,
                media=chunk,
            )

    # --------------------------------------------------------
    # GIF/video cannot be safely mixed with normal photo
    # InputMediaPhoto items, so send those individually.
    # --------------------------------------------------------

    caption_used = bool(
        photos
        and album_caption
    )

    for (
        msg,
        media_bytes,
        media_type,
    ) in non_photos:

        filename = get_filename(
            msg
        )

        if media_type == "animation":

            if not filename.lower().endswith(
                ".gif"
            ):

                filename = (
                    "original.gif"
                )

            file = BufferedInputFile(
                media_bytes,
                filename=filename,
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

            await bot.send_animation(
                chat_id=target_id,
                animation=file,
                caption=caption,
            )

        elif media_type == "video":

            if not any(
                filename.lower().endswith(
                    extension
                )
                for extension in (
                    ".mp4",
                    ".mov",
                    ".m4v",
                    ".webm",
                )
            ):

                filename = (
                    "original.mp4"
                )

            file = BufferedInputFile(
                media_bytes,
                filename=filename,
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

            await bot.send_video(
                chat_id=target_id,
                video=file,
                caption=caption,
            )

        else:

            file = BufferedInputFile(
                media_bytes,
                filename=(
                    filename
                    or "original.media"
                ),
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

            await bot.send_document(
                chat_id=target_id,
                document=file,
                caption=caption,
            )

    logger.info(
        "✅ ORIGINAL album reposted."
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
        # Group albums.
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
                item.id
                for item in group
            ),
        )

        # ----------------------------------------------------
        # Process groups.
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

                ids = [
                    item.id
                    for item in group
                ]

                async with processing_lock:

                    if any(
                        item_id
                        in processing_ids
                        for item_id in ids
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

                        for item_id in ids:

                            processing_ids.discard(
                                item_id
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
                    "❌ [%s] Processing crashed.",
                    msg.id,
                )

            finally:

                async with processing_lock:

                    processing_ids.discard(
                        msg.id
                    )

            await asyncio.sleep(
                1
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
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    logger.info(
        "🚀 Starting Telegram Repost Bot"
    )

    logger.info(
        "   🖼️ Original media: ENABLED"
    )

    logger.info(
        "   🤖 OpenAI generation: DISABLED"
    )

    logger.info(
        "   🧹 Watermark removal: DISABLED"
    )

    logger.info(
        "   🔁 Caption replacement: ENABLED"
    )

    logger.info(
        "   📚 Telegram albums: ENABLED"
    )

    logger.info(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    # --------------------------------------------------------
    # Telegram user client.
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
                    "📌 Last message in %s: %s",
                    source,
                    msg.id,
                )

                break

        except Exception:

            logger.exception(
                "❌ Could not get newest message "
                "from %s.",
                source,
            )

    # --------------------------------------------------------
    # Start polling.
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
