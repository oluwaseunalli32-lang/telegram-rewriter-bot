import os
import io
import asyncio
import logging
from collections import defaultdict

from dotenv import load_dotenv

# Load .env BEFORE importing modules that read environment variables.
load_dotenv()

from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.tl.types import MessageMediaPhoto, DocumentAttributeFilename

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BufferedInputFile, InputMediaPhoto

import database
from ai_processor import (
    rewrite_text,
    remove_watermarks_from_bytes,
    remove_watermarks_from_video_bytes,
    remove_watermarks_from_gif_bytes,
    generate_variations_from_clean_image,
    GENERATE_VARIATIONS,
    VARIATION_COUNT,
)

# ============================================================
# ENVIRONMENT
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_ID_RAW = os.getenv("API_ID", "").strip()
API_HASH = os.getenv("API_HASH", "").strip()
TELEGRAM_SESSION = os.getenv("TELEGRAM_SESSION", "").strip()

# New multi-channel setting.
# Example:
# SOURCE_CHANNELS=-1001111111111,-1002222222222,-1003333333333
#
# Backward-compatible: if SOURCE_CHANNELS is not set, the old
# SOURCE_CHANNEL value is used.
SOURCE_CHANNELS_RAW = os.getenv("SOURCE_CHANNELS", "").strip()

if SOURCE_CHANNELS_RAW:
    SOURCE_CHANNELS = []
    for value in SOURCE_CHANNELS_RAW.replace(";", ",").replace("\n", ",").split(","):
        value = value.strip()
        if not value:
            continue
        try:
            channel_id = int(value)
        except ValueError:
            raise RuntimeError(
                f"Invalid channel ID in SOURCE_CHANNELS: {value!r}"
            )
        if channel_id not in SOURCE_CHANNELS:
            SOURCE_CHANNELS.append(channel_id)
else:
    SOURCE_CHANNELS = [
        int(os.getenv("SOURCE_CHANNEL", "-1003593544389"))
    ]

TARGET_CHANNEL = int(
    os.getenv("TARGET_CHANNEL", "-1004415621706")
)
POLL_INTERVAL = max(
    1,
    int(os.getenv("POLL_INTERVAL", "3")),
)

try:
    API_ID = int(API_ID_RAW)
except ValueError:
    API_ID = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

if not TELEGRAM_SESSION:
    user_client = None
else:
    logger.info("🔐 Using Telegram StringSession.")
    user_client = TelegramClient(
        StringSession(TELEGRAM_SESSION),
        API_ID,
        API_HASH,
    )

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=None),
)

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
# VALIDATION
# ============================================================

def validate_environment():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is missing.")
    if not API_ID:
        raise RuntimeError("API_ID is missing or invalid.")
    if not API_HASH:
        raise RuntimeError("API_HASH is missing.")
    if not TELEGRAM_SESSION:
        raise RuntimeError("TELEGRAM_SESSION is missing.")
    if not SOURCE_CHANNELS:
        raise RuntimeError("SOURCE_CHANNELS is empty.")


# ============================================================
# MEDIA HELPERS
# ============================================================

def get_filename(message):
    if not message or not message.media:
        return "image.jpg"

    document = getattr(message, "document", None)
    if document:
        for attr in getattr(document, "attributes", []) or []:
            if isinstance(attr, DocumentAttributeFilename):
                return attr.file_name

    return "image.jpg"


def get_media_type(message):
    if isinstance(message.media, MessageMediaPhoto):
        return "photo"

    document = getattr(message, "document", None)
    if document:
        mime = (
            getattr(document, "mime_type", "") or ""
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
                if ext in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                }:
                    return "image"

    return "document"


async def download_media(message):
    buf = io.BytesIO()
    await user_client.download_media(
        message,
        file=buf,
    )
    return buf.getvalue()


# ============================================================
# CAPTION
# ============================================================

async def prepare_caption(message):
    caption = message.message or ""
    return await rewrite_text(caption) if caption else None


# ============================================================
# CLEAN + POST
# ============================================================

async def post_clean_outputs(
    clean_bytes,
    caption,
    filename,
):
    if not clean_bytes:
        return False

    outputs = [clean_bytes]

    if GENERATE_VARIATIONS:
        generated = await generate_variations_from_clean_image(
            clean_bytes,
            filename,
            VARIATION_COUNT,
        )
        if generated:
            outputs = generated

    for index, image_bytes in enumerate(outputs):
        out_name = (
            "clean.png"
            if len(outputs) == 1
            else f"clean_variation_{index + 1}.png"
        )

        await bot.send_photo(
            TARGET_CHANNEL,
            photo=BufferedInputFile(
                image_bytes,
                filename=out_name,
            ),
            caption=caption if index == 0 else None,
        )

    return True


async def send_single(message, source_channel):
    try:
        media_type = (
            get_media_type(message)
            if message.media
            else "text"
        )

        logger.info(
            "📨 New source message %s | source=%s | type=%s | file=%s",
            message.id,
            source_channel,
            media_type,
            get_filename(message),
        )

        if not message.media:
            caption = await prepare_caption(message)

            if caption:
                await bot.send_message(
                    TARGET_CHANNEL,
                    caption,
                )

            return True

        filename = get_filename(message)
        original = await download_media(message)

        if not original:
            logger.error(
                "❌ Download failed for %s from %s",
                message.id,
                source_channel,
            )
            return False

        caption = await prepare_caption(message)

        # --------------------------------------------------------
        # STILL IMAGE
        # --------------------------------------------------------
        if media_type in {"photo", "image"}:
            cleaned = await remove_watermarks_from_bytes(
                original,
                filename,
            )

            if cleaned:
                logger.info(
                    "🧼 Clean still generated for %s from %s",
                    message.id,
                    source_channel,
                )
                return await post_clean_outputs(
                    cleaned,
                    caption,
                    filename,
                )

            # No convincing watermark signal, or cleanup failed.
            logger.info(
                "📌 Posting original still for %s from %s",
                message.id,
                source_channel,
            )

            await bot.send_photo(
                TARGET_CHANNEL,
                photo=BufferedInputFile(
                    original,
                    filename=filename,
                ),
                caption=caption,
            )
            return True

        # --------------------------------------------------------
        # GIF / VIDEO -> ONE CLEAN STILL
        # --------------------------------------------------------
        if media_type == "animation":
            cleaned = await remove_watermarks_from_gif_bytes(
                original,
                filename,
            )
        elif media_type == "video":
            cleaned = await remove_watermarks_from_video_bytes(
                original,
                filename,
            )
        else:
            cleaned = None

        if media_type in {"animation", "video"}:
            if cleaned:
                logger.info(
                    "🧼 Motion converted to one clean still for %s from %s",
                    message.id,
                    source_channel,
                )
                return await post_clean_outputs(
                    cleaned,
                    caption,
                    filename,
                )

            # Do NOT repost GIF/video because the desired behavior is a
            # clean still output.
            logger.warning(
                "⚠️ Motion %s from %s could not be converted to a clean still; "
                "original motion is NOT reposted.",
                message.id,
                source_channel,
            )
            return False

        # --------------------------------------------------------
        # OTHER DOCUMENT
        # --------------------------------------------------------
        await bot.send_document(
            TARGET_CHANNEL,
            document=BufferedInputFile(
                original,
                filename=filename,
            ),
            caption=caption,
        )
        return True

    except Exception:
        logger.exception(
            "❌ Failed processing message %s from source %s",
            message.id,
            source_channel,
        )
        return False


async def process_album(messages, source_channel):
    messages = sorted(
        messages,
        key=lambda m: m.id,
    )

    caption = None

    for msg in messages:
        if msg.message:
            caption = await rewrite_text(msg.message)
            break

    photo_outputs = []

    for msg in messages:
        if not msg.media:
            continue

        filename = get_filename(msg)
        media_type = get_media_type(msg)
        original = await download_media(msg)

        if not original:
            logger.error(
                "❌ Album item %s from %s could not be downloaded.",
                msg.id,
                source_channel,
            )
            return False

        if media_type in {"photo", "image"}:
            cleaned = await remove_watermarks_from_bytes(
                original,
                filename,
            )
            final = cleaned or original
            photo_outputs.append((final, filename))

        elif media_type == "animation":
            cleaned = await remove_watermarks_from_gif_bytes(
                original,
                filename,
            )

            if not cleaned:
                return False

            await post_clean_outputs(
                cleaned,
                caption,
                filename,
            )
            caption = None

        elif media_type == "video":
            cleaned = await remove_watermarks_from_video_bytes(
                original,
                filename,
            )

            if not cleaned:
                return False

            await post_clean_outputs(
                cleaned,
                caption,
                filename,
            )
            caption = None

        else:
            await bot.send_document(
                TARGET_CHANNEL,
                document=BufferedInputFile(
                    original,
                    filename=filename,
                ),
                caption=caption,
            )
            caption = None

    if photo_outputs:
        for start in range(0, len(photo_outputs), 10):
            chunk = photo_outputs[start:start + 10]
            group = []

            for idx, (image_bytes, filename) in enumerate(chunk):
                group.append(
                    InputMediaPhoto(
                        media=BufferedInputFile(
                            image_bytes,
                            filename="clean.png",
                        ),
                        caption=caption if idx == 0 else None,
                    )
                )

            await bot.send_media_group(
                TARGET_CHANNEL,
                media=group,
            )
            caption = None

    return True


# ============================================================
# DATABASE STATE -- PER SOURCE CHANNEL
# ============================================================

def get_last_processed(source_channel):
    try:
        return database.get_last_processed(
            source_channel,
        )
    except Exception:
        logger.exception(
            "⚠️ Could not read last processed message "
            "for source %s.",
            source_channel,
        )
        return None


def set_last_processed(source_channel, message_id):
    try:
        return database.set_last_processed(
            source_channel,
            message_id,
        )
    except Exception:
        logger.exception(
            "⚠️ Could not save last processed message ID %s "
            "for source %s.",
            message_id,
            source_channel,
        )
        return False


# ============================================================
# ONE SOURCE CHANNEL POLLER
# ============================================================

async def process_channel(source_channel):
    last_id = get_last_processed(source_channel)

    # --------------------------------------------------------
    # FIRST RUN FOR THIS SOURCE
    # --------------------------------------------------------
    if last_id is None:
        latest = await user_client.get_messages(
            source_channel,
            limit=1,
        )

        if latest:
            last_id = latest[0].id
            set_last_processed(
                source_channel,
                last_id,
            )

            logger.info(
                "📌 Source %s initial position set to message %s. "
                "Existing posts will not be replayed.",
                source_channel,
                last_id,
            )

    logger.info(
        "📡 Polling source %s starting after %s",
        source_channel,
        last_id or 0,
    )

    # --------------------------------------------------------
    # POLLING LOOP
    # --------------------------------------------------------
    while True:
        try:
            messages = await user_client.get_messages(
                source_channel,
                min_id=last_id or 0,
                limit=100,
                reverse=True,
            )

            if messages:
                logger.info(
                    "📨 Found %d new source message(s) in %s.",
                    len(messages),
                    source_channel,
                )

                groups = defaultdict(list)
                normal = []

                for msg in messages:
                    if msg.grouped_id:
                        groups[msg.grouped_id].append(msg)
                    else:
                        normal.append(msg)

                # Preserve chronological order across normal messages
                # and albums.
                work = [
                    (
                        min(m.id for m in group),
                        "album",
                        group,
                    )
                    for group in groups.values()
                ]

                work += [
                    (
                        msg.id,
                        "single",
                        msg,
                    )
                    for msg in normal
                ]

                work.sort(
                    key=lambda item: item[0]
                )

                for _, kind, payload in work:
                    if kind == "single":
                        msg = payload

                        logger.info(
                            "➡️ Processing source=%s message=%s (%s)",
                            source_channel,
                            msg.id,
                            get_media_type(msg)
                            if msg.media
                            else "text",
                        )

                        ok = await send_single(
                            msg,
                            source_channel,
                        )

                        if ok:
                            last_id = max(
                                last_id or 0,
                                msg.id,
                            )
                            set_last_processed(
                                source_channel,
                                last_id,
                            )
                        else:
                            logger.warning(
                                "⏸️ Source %s message %s remains pending "
                                "because processing/posting failed.",
                                source_channel,
                                msg.id,
                            )
                            break

                    else:
                        group = payload
                        first = min(
                            m.id
                            for m in group
                        )

                        logger.info(
                            "➡️ Processing source=%s album starting at message %s (%d items)",
                            source_channel,
                            first,
                            len(group),
                        )

                        ok = await process_album(
                            group,
                            source_channel,
                        )

                        if ok:
                            last_id = max(
                                last_id or 0,
                                max(m.id for m in group),
                            )
                            set_last_processed(
                                source_channel,
                                last_id,
                            )
                        else:
                            logger.warning(
                                "⏸️ Source %s album starting at %s remains pending "
                                "because processing/posting failed.",
                                source_channel,
                                first,
                            )
                            break

            await asyncio.sleep(POLL_INTERVAL)

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "❌ Polling loop error for source %s",
                source_channel,
            )
            await asyncio.sleep(POLL_INTERVAL)


# ============================================================
# STARTUP
# ============================================================

async def main():
    validate_environment()

    logger.info(
        "🚀 Starting Telegram Image Recreation Bot..."
    )
    logger.info(
        "📌 Sources (%d): %s",
        len(SOURCE_CHANNELS),
        ", ".join(str(x) for x in SOURCE_CHANNELS),
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
        "🖼️ Still images: location-independent watermark search"
    )
    logger.info(
        "🎞️ GIF/MP4: representative frame -> ONE clean still image"
    )
    logger.info(
        "🚫 GIF/MP4 will NOT be reposted as animation"
    )
    logger.info(
        "📝 Caption: remove '*' + replace @cappersfree only"
    )
    logger.info(
        "♾️ OpenAI application image limit: NONE"
    )
    logger.info(
        "🧩 Four variations: %s (count=%d)",
        "ON" if GENERATE_VARIATIONS else "OFF",
        VARIATION_COUNT,
    )
    logger.info(
        "🎨 OpenAI quality: %s",
        os.getenv("OPENAI_IMAGE_QUALITY", "low"),
    )
    logger.info(
        "🤖 OpenAI model: %s",
        os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
    )

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError(
            "OPENAI_API_KEY is missing. "
            "This production version requires OpenAI for cleanup."
        )

    if user_client is None:
        raise RuntimeError(
            "Telegram user client could not be created because "
            "TELEGRAM_SESSION is missing."
        )

    await user_client.connect()

    if not await user_client.is_user_authorized():
        await user_client.disconnect()
        raise RuntimeError(
            "Telegram StringSession is not authorized."
        )

    me = await user_client.get_me()

    logger.info(
        "✅ Telegram account authenticated: %s | @%s",
        me.first_name or "",
        me.username or "",
    )
    logger.info(
        "✅ Telegram user client connected!"
    )

    # --------------------------------------------------------
    # VERIFY EVERY SOURCE AND TARGET
    # --------------------------------------------------------
    for source_channel in SOURCE_CHANNELS:
        source_entity = await user_client.get_entity(
            source_channel
        )
        logger.info(
            "✅ Source resolved: %s | %s",
            source_channel,
            getattr(
                source_entity,
                "title",
                None,
            ) or source_channel,
        )

    target_entity = await user_client.get_entity(
        TARGET_CHANNEL
    )
    logger.info(
        "✅ Target resolved: %s | %s",
        TARGET_CHANNEL,
        getattr(
            target_entity,
            "title",
            None,
        ) or TARGET_CHANNEL,
    )

    # --------------------------------------------------------
    # RUN ALL SOURCE POLLERS CONCURRENTLY
    # Each source has its own database position.
    # --------------------------------------------------------
    try:
        await asyncio.gather(
            *(
                process_channel(source_channel)
                for source_channel in SOURCE_CHANNELS
            )
        )

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
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped.")

    except Exception:
        logger.exception("💥 Fatal startup error.")
        raise
