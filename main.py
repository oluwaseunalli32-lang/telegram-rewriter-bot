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
    generate_variations_from_clean_image,
)

load_dotenv()

# ============================================================
# ENVIRONMENT
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_ID_RAW = os.getenv("API_ID", "").strip()
API_HASH = os.getenv("API_HASH", "").strip()
PHONE_NUMBER = os.getenv("PHONE_NUMBER", "").strip()
TELEGRAM_SESSION = os.getenv("TELEGRAM_SESSION", "").strip()

SOURCE_CHANNEL = int(os.getenv("SOURCE_CHANNEL", "-1003593544389"))
TARGET_CHANNEL = int(os.getenv("TARGET_CHANNEL", "-1004415621706"))
POLL_INTERVAL = max(1, int(os.getenv("POLL_INTERVAL", "3")))

BETA_MODE = os.getenv("BETA_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
GENERATE_VARIATIONS = os.getenv("GENERATE_VARIATIONS", "false").strip().lower() in {"1", "true", "yes", "on"}
VARIATION_COUNT = max(1, min(4, int(os.getenv("VARIATION_COUNT", "4"))))

try:
    API_ID = int(API_ID_RAW)
except ValueError:
    API_ID = 0

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# CLIENTS
# ============================================================

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
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".mov", ".m4v", ".webm",
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


# ============================================================
# AI PROCESSING
# ============================================================

async def clean_still(original_bytes, filename):
    return await remove_watermarks_from_bytes(
        original_bytes,
        filename,
    )


async def clean_motion(original_bytes, filename, media_type):
    if media_type == "animation":
        return await remove_watermarks_from_gif_bytes(
            original_bytes,
            filename,
        )

    return await remove_watermarks_from_video_bytes(
        original_bytes,
        filename,
    )


# ============================================================
# CAPTION
# ============================================================

async def prepare_caption(message):
    caption = message.message or ""
    if not caption:
        return None
    return await rewrite_text(caption)


# ============================================================
# POST ONE PHOTO/IMAGE
# ============================================================

async def send_clean_image_outputs(
    clean_bytes,
    caption,
    filename,
):
    """Post one cleaned image, or optionally four variations."""

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
        file = BufferedInputFile(
            image_bytes,
            filename=(
                f"clean_{index + 1}.png"
                if len(outputs) > 1
                else "clean.png"
            ),
        )

        await bot.send_photo(
            TARGET_CHANNEL,
            photo=file,
            caption=caption if index == 0 else None,
        )

    return True


# ============================================================
# SEND SINGLE MESSAGE
# ============================================================

async def send_single(message):
    try:
        if not message.media:
            caption = await prepare_caption(message)
            if caption:
                await bot.send_message(TARGET_CHANNEL, caption)
            logger.info("✅ Text message %s posted.", message.id)
            return True

        filename = get_filename(message)
        media_type = get_media_type(message)

        logger.info(
            "📨 New source message %s | type=%s | file=%s",
            message.id,
            media_type,
            filename,
        )

        original = await download_media(message)
        if not original:
            logger.error("❌ Media download failed for %s", message.id)
            return False

        caption = await prepare_caption(message)

        # ----------------------------------------------------
        # STILL IMAGE
        # ----------------------------------------------------
        if media_type in {"photo", "image"}:
            cleaned = await clean_still(original, filename)

            if cleaned:
                logger.info("🧼 Clean still generated for %s", message.id)
                return await send_clean_image_outputs(
                    cleaned,
                    caption,
                    filename,
                )

            # No confident watermark signal OR AI failure:
            # preserve the source rather than lose the post.
            logger.info("📌 Using original still for %s", message.id)
            file = BufferedInputFile(original, filename=filename)
            await bot.send_photo(
                TARGET_CHANNEL,
                photo=file,
                caption=caption,
            )
            return True

        # ----------------------------------------------------
        # GIF / MP4 -> CLEAN STILL IMAGE
        # ----------------------------------------------------
        if media_type in {"animation", "video"}:
            cleaned = await clean_motion(
                original,
                filename,
                media_type,
            )

            if cleaned:
                logger.info(
                    "🧼 Motion converted to one clean still for %s",
                    message.id,
                )
                return await send_clean_image_outputs(
                    cleaned,
                    caption,
                    filename,
                )

            # If AI could not clean the motion, do NOT repost the GIF/MP4.
            # This keeps the target free of the pulsing watermark animation.
            logger.warning(
                "⚠️ Could not create a clean still for motion %s; not reposting the GIF/video.",
                message.id,
            )
            return False

        # ----------------------------------------------------
        # OTHER DOCUMENTS
        # ----------------------------------------------------
        file = BufferedInputFile(original, filename=filename)
        await bot.send_document(
            TARGET_CHANNEL,
            document=file,
            caption=caption,
        )

        logger.info("✅ Document %s posted.", message.id)
        return True

    except Exception:
        logger.exception("❌ Failed processing message %s", message.id)
        return False


# ============================================================
# ALBUM
# ============================================================

async def process_album(messages):
    if not messages:
        return False

    messages = sorted(messages, key=lambda m: m.id)

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
            logger.error("❌ Could not download album item %s", msg.id)
            continue

        if media_type in {"photo", "image"}:
            cleaned = await clean_still(original, filename)
            final_bytes = cleaned if cleaned else original

            # Album photos remain grouped when no variation stage is enabled.
            # With 4 variations, send each resulting image separately because
            # the same caption/content should not be duplicated across a group.
            if GENERATE_VARIATIONS and cleaned:
                outputs = await generate_variations_from_clean_image(
                    cleaned,
                    filename,
                    VARIATION_COUNT,
                )
                for out in outputs:
                    await bot.send_photo(
                        TARGET_CHANNEL,
                        photo=BufferedInputFile(out, filename="clean.png"),
                        caption=caption,
                    )
                    caption = None
            else:
                photo_outputs.append((final_bytes, filename))

        elif media_type in {"animation", "video"}:
            cleaned = await clean_motion(
                original,
                filename,
                media_type,
            )

            if cleaned:
                await bot.send_photo(
                    TARGET_CHANNEL,
                    photo=BufferedInputFile(cleaned, filename="clean_motion.png"),
                    caption=caption,
                )
                caption = None
            else:
                logger.warning(
                    "⚠️ Album motion item %s could not be cleaned and was not posted.",
                    msg.id,
                )

        else:
            await bot.send_document(
                TARGET_CHANNEL,
                document=BufferedInputFile(original, filename=filename),
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
# DATABASE STATE
# ============================================================

def get_last_processed():
    try:
        return database.get_last_processed(SOURCE_CHANNEL)
    except Exception:
        logger.exception("⚠️ Could not read last processed message from database.")
        return None


def set_last_processed(message_id):
    try:
        return database.set_last_processed(
            SOURCE_CHANNEL,
            message_id,
        )
    except Exception:
        logger.exception(
            "⚠️ Could not save last processed message ID %s.",
            message_id,
        )
        return False


# ============================================================
# CHANNEL POLLING
# ============================================================

async def process_channel():
    last_id = get_last_processed()

    if last_id is None:
        latest = await user_client.get_messages(
            SOURCE_CHANNEL,
            limit=1,
        )

        if latest:
            last_id = latest[0].id
            set_last_processed(last_id)
            logger.info(
                "📌 Initial position set to message %s. Existing posts will not be replayed.",
                last_id,
            )

    logger.info(
        "📡 Polling source channel starting after %s",
        last_id or 0,
    )

    while True:
        try:
            messages = await user_client.get_messages(
                SOURCE_CHANNEL,
                min_id=last_id or 0,
                limit=100,
                reverse=True,
            )

            if messages:
                logger.info(
                    "📨 Found %d new source message(s).",
                    len(messages),
                )

                album_groups = defaultdict(list)
                normal = []

                for msg in messages:
                    if msg.grouped_id:
                        album_groups[msg.grouped_id].append(msg)
                    else:
                        normal.append(msg)

                for msg in normal:
                    logger.info(
                        "➡️ Processing new message %s (%s)",
                        msg.id,
                        get_media_type(msg) if msg.media else "text",
                    )

                    if await send_single(msg):
                        last_id = max(last_id or 0, msg.id)
                        set_last_processed(last_id)
                    else:
                        # Do not mark failed messages as processed.
                        logger.warning(
                            "⏸️ Message %s was NOT marked processed because posting failed.",
                            msg.id,
                        )

                for _, group in sorted(
                    album_groups.items(),
                    key=lambda kv: min(m.id for m in kv[1]),
                ):
                    first_id = min(m.id for m in group)
                    logger.info(
                        "➡️ Processing album starting at message %s (%d items)",
                        first_id,
                        len(group),
                    )

                    if await process_album(group):
                        last_id = max(last_id or 0, max(m.id for m in group))
                        set_last_processed(last_id)
                    else:
                        logger.warning(
                            "⏸️ Album was not marked processed because posting failed."
                        )

            await asyncio.sleep(POLL_INTERVAL)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("❌ Polling loop error")
            await asyncio.sleep(POLL_INTERVAL)


# ============================================================
# STARTUP
# ============================================================

async def main():
    validate_environment()

    logger.info("🚀 Starting Telegram Image Recreation Bot...")
    logger.info("📌 Source: %s", SOURCE_CHANNEL)
    logger.info("📌 Target: %s", TARGET_CHANNEL)
    logger.info("🔐 Telegram authentication: StringSession")
    logger.info("🧹 Watermark removal: ENABLED")
    logger.info("🖼️ Still images: location-independent watermark search")
    logger.info("🎞️ GIF/MP4: representative frame -> ONE clean still image")
    logger.info("🚫 GIF/MP4 will NOT be reposted as animation")
    logger.info("📝 Caption: remove '*' + replace @cappersfree only")
    logger.info("🧪 Beta mode: %s", BETA_MODE)
    logger.info(
        "💰 Max OpenAI image outputs per run: %s",
        os.getenv("MAX_OPENAI_IMAGES_PER_RUN", "1"),
    )
    logger.info(
        "🧩 Four variations feature: %s (count=%s)",
        "ON" if GENERATE_VARIATIONS else "OFF",
        VARIATION_COUNT,
    )

    if os.getenv("OPENAI_API_KEY", "").strip():
        logger.info("✅ OPENAI_API_KEY detected.")
        logger.info(
            "🎨 OpenAI image quality: %s",
            os.getenv("OPENAI_IMAGE_QUALITY", "low"),
        )
        logger.info(
            "🤖 OpenAI image model: %s",
            os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
        )
    else:
        logger.warning(
            "⚠️ OPENAI_API_KEY missing; still images will be reposted unchanged and motion items will not be reposted."
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
    logger.info("✅ Telegram user client connected!")

    # Verify both entities are accessible before entering the loop.
    try:
        source_entity = await user_client.get_entity(SOURCE_CHANNEL)
        target_entity = await user_client.get_entity(TARGET_CHANNEL)

        logger.info(
            "✅ Source resolved: %s",
            getattr(source_entity, "title", None)
            or getattr(source_entity, "username", None)
            or SOURCE_CHANNEL,
        )
        logger.info(
            "✅ Target resolved: %s",
            getattr(target_entity, "title", None)
            or getattr(target_entity, "username", None)
            or TARGET_CHANNEL,
        )
    except Exception:
        logger.exception("❌ Could not resolve source/target channel. Check account access and IDs.")
        await user_client.disconnect()
        raise

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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped.")
    except Exception:
        logger.exception("💥 Fatal startup error.")
        raise
