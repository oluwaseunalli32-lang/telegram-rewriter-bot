import os
import io
import asyncio
import logging
from collections import defaultdict

from dotenv import load_dotenv

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
)

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_ID_RAW = os.getenv("API_ID", "").strip()
API_HASH = os.getenv("API_HASH", "").strip()
PHONE_NUMBER = os.getenv("PHONE_NUMBER", "").strip()
TELEGRAM_SESSION = os.getenv("TELEGRAM_SESSION", "").strip()

SOURCE_CHANNEL = int(os.getenv("SOURCE_CHANNEL", "-1003593544389"))
TARGET_CHANNEL = int(os.getenv("TARGET_CHANNEL", "-1004415621706"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

try:
    API_ID = int(API_ID_RAW)
except ValueError:
    API_ID = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

user_client = TelegramClient(StringSession(TELEGRAM_SESSION), API_ID, API_HASH)
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov", ".m4v", ".webm"}


def validate_environment():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is missing.")
    if not API_ID:
        raise RuntimeError("API_ID is missing or invalid.")
    if not API_HASH:
        raise RuntimeError("API_HASH is missing.")
    if not TELEGRAM_SESSION:
        raise RuntimeError("TELEGRAM_SESSION is missing.")


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
        mime = (getattr(document, "mime_type", "") or "").lower()
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
                if ext in {".mp4", ".mov", ".m4v", ".webm"}:
                    return "video"
                if ext in {".jpg", ".jpeg", ".png", ".webp"}:
                    return "image"
    return "document"


async def download_media(message):
    buf = io.BytesIO()
    await user_client.download_media(message, file=buf)
    return buf.getvalue()


async def process_still(media_bytes, filename):
    cleaned = await remove_watermarks_from_bytes(media_bytes, filename)
    return cleaned if cleaned else None


async def process_motion(media_bytes, filename):
    # Beta: local processing only. No OpenAI cost.
    cleaned = remove_watermarks_from_video_bytes(media_bytes, filename)
    return cleaned if cleaned else None


async def prepare_caption(message):
    caption = message.message or ""
    return await rewrite_text(caption) if caption else None


async def send_single(message):
    try:
        if not message.media:
            caption = await prepare_caption(message)
            if caption:
                await bot.send_message(TARGET_CHANNEL, caption)
            return True

        filename = get_filename(message)
        media_type = get_media_type(message)
        original = await download_media(message)
        if not original:
            return False

        caption = await prepare_caption(message)
        final_bytes = original

        if media_type in {"photo", "image"}:
            cleaned = await process_still(original, filename)
            if cleaned:
                final_bytes = cleaned

        elif media_type == "animation":
            cleaned = remove_watermarks_from_gif_bytes(original, filename)
            if cleaned:
                final_bytes = cleaned

        elif media_type == "video":
            cleaned = await process_motion(original, filename)
            if cleaned:
                final_bytes = cleaned

        file = BufferedInputFile(final_bytes, filename=filename)

        if media_type in {"photo", "image"}:
            await bot.send_photo(TARGET_CHANNEL, photo=file, caption=caption)
        elif media_type == "animation":
            await bot.send_animation(TARGET_CHANNEL, animation=file, caption=caption)
        elif media_type == "video":
            await bot.send_video(TARGET_CHANNEL, video=file, caption=caption)
        else:
            await bot.send_document(TARGET_CHANNEL, document=file, caption=caption)

        logger.info("✅ Message %s posted.", message.id)
        return True

    except Exception:
        logger.exception("❌ Failed processing message %s", message.id)
        return False


async def process_album(messages):
    if not messages:
        return False

    messages = sorted(messages, key=lambda m: m.id)
    caption = None
    for msg in messages:
        if msg.message:
            caption = await rewrite_text(msg.message)
            break

    photos = []

    for msg in messages:
        if not msg.media:
            continue

        filename = get_filename(msg)
        media_type = get_media_type(msg)
        original = await download_media(msg)
        if not original:
            continue

        if media_type in {"photo", "image"}:
            cleaned = await process_still(original, filename)
            final_bytes = cleaned if cleaned else original
            photos.append((final_bytes, filename))
        else:
            if media_type == "animation":
                cleaned = remove_watermarks_from_gif_bytes(original, filename)
            elif media_type == "video":
                cleaned = await process_motion(original, filename)
            else:
                cleaned = None
            final_bytes = cleaned if cleaned else original
            file = BufferedInputFile(final_bytes, filename=filename)

            if media_type == "animation":
                await bot.send_animation(TARGET_CHANNEL, animation=file, caption=caption)
            elif media_type == "video":
                await bot.send_video(TARGET_CHANNEL, video=file, caption=caption)
            else:
                await bot.send_document(TARGET_CHANNEL, document=file, caption=caption)
            caption = None

    if photos:
        for start in range(0, len(photos), 10):
            chunk = photos[start:start + 10]
            group = []
            for idx, (image_bytes, filename) in enumerate(chunk):
                file = BufferedInputFile(image_bytes, filename=filename)
                group.append(InputMediaPhoto(media=file, caption=caption if idx == 0 else None))
            await bot.send_media_group(TARGET_CHANNEL, media=group)
            caption = None

    return True


def get_last_processed():
    try:
        return database.get_last_processed(SOURCE_CHANNEL)
    except Exception:
        logger.exception("⚠️ Could not read last processed message from database.")
        return None


def set_last_processed(message_id):
    try:
        return database.set_last_processed(SOURCE_CHANNEL, message_id)
    except Exception:
        logger.exception("⚠️ Could not save last processed message ID %s.", message_id)
        return False


async def process_channel():
    last_id = get_last_processed()

    if last_id is None:
        latest = await user_client.get_messages(SOURCE_CHANNEL, limit=1)
        if latest:
            last_id = latest[0].id
            set_last_processed(last_id)
            logger.info("📌 Initial position set to message %s. Existing posts will not be replayed.", last_id)

    logger.info("📡 Polling source channel starting after %s", last_id or 0)

    while True:
        try:
            messages = await user_client.get_messages(
                SOURCE_CHANNEL,
                min_id=last_id or 0,
                limit=100,
                reverse=True,
            )

            if messages:
                album_groups = defaultdict(list)
                normal = []

                for msg in messages:
                    if msg.grouped_id:
                        album_groups[msg.grouped_id].append(msg)
                    else:
                        normal.append(msg)

                for msg in normal:
                    if await send_single(msg):
                        last_id = max(last_id or 0, msg.id)
                        set_last_processed(last_id)

                for _, group in sorted(album_groups.items(), key=lambda kv: min(m.id for m in kv[1])):
                    if await process_album(group):
                        last_id = max(last_id or 0, max(m.id for m in group))
                        set_last_processed(last_id)

            await asyncio.sleep(POLL_INTERVAL)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("❌ Polling loop error")
            await asyncio.sleep(POLL_INTERVAL)


async def main():
    validate_environment()

    logger.info("🚀 Starting Telegram Image Recreation Bot...")
    logger.info("📌 Source: %s", SOURCE_CHANNEL)
    logger.info("📌 Target: %s", TARGET_CHANNEL)
    logger.info("🔐 Telegram authentication: StringSession")
    logger.info("🧪 BETA MODE: conservative testing enabled")
    logger.info("💰 Max OpenAI still-image edits per process run: %s", os.getenv("MAX_OPENAI_EDITS_PER_RUN", "1"))
    logger.info("🎞️ GIF/MP4: local frame processing; no OpenAI")
    logger.info("📝 Caption handling: remove '*' + exact username replacement")

    if os.getenv("OPENAI_API_KEY", "").strip():
        logger.info("✅ OPENAI_API_KEY detected.")
    else:
        logger.warning("⚠️ OPENAI_API_KEY missing; still images stay original.")

    await user_client.connect()

    if not await user_client.is_user_authorized():
        await user_client.disconnect()
        raise RuntimeError("Telegram StringSession is not authorized.")

    me = await user_client.get_me()
    logger.info("✅ Telegram account authenticated: %s | @%s", me.first_name or "", me.username or "")
    logger.info("✅ Telegram user client connected!")

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


if __name__ == "__main__":
    asyncio.run(main())
