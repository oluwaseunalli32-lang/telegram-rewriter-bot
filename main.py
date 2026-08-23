import os
import asyncio
import logging
from pathlib import Path


# ============================================================
# LOAD ENVIRONMENT FIRST
# ============================================================

from dotenv import load_dotenv


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

from aiogram import Bot
from telethon import (
    TelegramClient,
    errors,
)

import database

from ai_processor import (
    rewrite_text,
    regenerate_image_from_bytes,
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

BOT_TOKEN = (
    os.getenv(
        "BOT_TOKEN",
        "",
    ).strip()
)

API_ID_RAW = (
    os.getenv(
        "API_ID",
        "",
    ).strip()
)

API_HASH = (
    os.getenv(
        "API_HASH",
        "",
    ).strip()
)

PHONE = (
    os.getenv(
        "PHONE_NUMBER",
        "",
    ).strip()
)


try:

    API_ID = int(
        API_ID_RAW
    )

except ValueError:

    API_ID = 0


# ============================================================
# VALIDATE ENVIRONMENT
# ============================================================

missing = []

if not BOT_TOKEN:
    missing.append(
        "BOT_TOKEN"
    )

if not API_ID:
    missing.append(
        "API_ID"
    )

if not API_HASH:
    missing.append(
        "API_HASH"
    )

if not PHONE:
    missing.append(
        "PHONE_NUMBER"
    )


if missing:

    logger.error(
        "❌ Missing environment variables: %s",
        ", ".join(missing),
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
# DEFAULT CHANNEL CONFIGURATION
# ============================================================

SOURCE_CHANNEL_ID = -1003593544389

TARGET_CHANNEL_ID = -1004415621706


# ============================================================
# PROCESSING STATE
# ============================================================

# Prevent the same message from being processed twice
# during this process lifetime.

processing_ids = set()

processing_lock = asyncio.Lock()


# Last successfully handled message per source.

last_processed = {}


# ============================================================
# TELEGRAM CONNECTION
# ============================================================

async def ensure_connection():
    """
    Ensure Telethon is connected.
    """

    if user_client.is_connected():

        return True

    logger.warning(
        "⚠️ Telegram client disconnected. "
        "Attempting reconnect..."
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
            "❌ Telegram reconnection failed: %s",
            e,
        )

        return False


# ============================================================
# MEDIA DETECTION
# ============================================================

def get_image_media(msg):
    """
    Return Telegram media that can be processed.

    Supports:
        photo
        image/*
        video/*
        GIF
        MP4
        MOV
        WEBM
        common image extensions
    """

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

        # Image MIME.

        if mime_type.startswith(
            "image/"
        ):

            return document

        # Video MIME.

        if mime_type.startswith(
            "video/"
        ):

            return document

        # Extension fallback.

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
            ".webm",
        }

        for attribute in attributes:

            file_name = (
                getattr(
                    attribute,
                    "file_name",
                    "",
                )
                or ""
            )

            extension = (
                Path(
                    file_name
                )
                .suffix
                .lower()
            )

            if extension in (
                supported_extensions
            ):

                return document

    # --------------------------------------------------------
    # Generic media fallback.
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
# PROCESS SINGLE IMAGE / GIF
# ============================================================

async def process_single_media(
    media,
    caption_text,
    target_id,
    message_id,
):
    """
    Process one image/GIF:

        Telegram
            ↓
        Download
            ↓
        Vision
            ↓
        Deconstruction
            ↓
        Image generation
            ↓
        NEW IMAGE
            ↓
        Target Telegram channel

    Original media is never posted if generation fails.
    """

    try:

        # ----------------------------------------------------
        # Download.
        # ----------------------------------------------------

        logger.info(
            "🖼️ [%s] Downloading original image/GIF...",
            message_id,
        )

        image_bytes = (
            await user_client.download_media(
                media,
                bytes,
            )
        )

        if not image_bytes:

            logger.error(
                "❌ [%s] Could not download media.",
                message_id,
            )

            return False

        logger.info(
            "📦 [%s] Downloaded %d bytes",
            message_id,
            len(image_bytes),
        )

        try:

            logger.info(
                "🔬 [%s] Media header: %s",
                message_id,
                bytes(
                    image_bytes[:32]
                ).hex(" "),
            )

        except Exception:
            pass

        # ----------------------------------------------------
        # Regenerate.
        # ----------------------------------------------------

        logger.info(
            "🔍 [%s] Starting image recreation:",
            message_id,
        )

        logger.info(
            "    ORIGINAL"
        )

        logger.info(
            "       ↓"
        )

        logger.info(
            "    OPENAI VISION"
        )

        logger.info(
            "       ↓"
        )

        logger.info(
            "    DECONSTRUCTION"
        )

        logger.info(
            "       ↓"
        )

        logger.info(
            "    IMAGE GENERATION"
        )

        logger.info(
            "       ↓"
        )

        logger.info(
            "    NEW IMAGE"
        )

        new_image_data = (
            await regenerate_image_from_bytes(
                image_bytes
            )
        )

        if not new_image_data:

            logger.error(
                "❌ [%s] Image regeneration failed. "
                "Original image will NOT be reposted.",
                message_id,
            )

            return False

        # ----------------------------------------------------
        # Send regenerated image.
        # ----------------------------------------------------

        logger.info(
            "📤 [%s] Sending NEW regenerated image...",
            message_id,
        )

        send_kwargs = {
            "chat_id": target_id,
            "photo": new_image_data,
        }

        if caption_text:

            send_kwargs[
                "caption"
            ] = caption_text[:1024]

        await bot.send_photo(
            **send_kwargs
        )

        logger.info(
            "✅ [%s] NEW regenerated image posted.",
            message_id,
        )

        return True

    except Exception as e:

        logger.exception(
            "❌ [%s] Error processing image: %s",
            message_id,
            e,
        )

        return False


# ============================================================
# PROCESS CHANNEL
# ============================================================

async def process_channel(
    source_id,
    target_id,
):
    """
    Process new messages from one source channel.
    """

    if not await ensure_connection():

        logger.error(
            "❌ Cannot process channel "
            "%s because Telegram is disconnected.",
            source_id,
        )

        return

    try:

        channel = (
            await user_client.get_entity(
                source_id
            )
        )

        last_id = (
            last_processed.get(
                source_id,
                0,
            )
        )

        logger.info(
            "🔎 Checking source %s after message %s",
            source_id,
            last_id,
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

        logger.info(
            "📩 Found %d new message(s) in %s",
            len(new_messages),
            source_id,
        )

        # ----------------------------------------------------
        # Process chronologically.
        # ----------------------------------------------------

        for msg in new_messages:

            # ------------------------------------------------
            # Duplicate protection.
            # ------------------------------------------------

            async with processing_lock:

                if msg.id in processing_ids:

                    logger.warning(
                        "⚠️ Message %s already processing. "
                        "Skipping duplicate.",
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
                    "📩 Processing message ID: %s",
                    msg.id,
                )

                # ------------------------------------------------
                # Caption / text.
                # ------------------------------------------------

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

                rewritten_text = (
                    await rewrite_text(
                        original_text
                    )
                )

                if (
                    original_text
                    != rewritten_text
                ):

                    logger.info(
                        "✏️ [%s] Username replacement applied.",
                        msg.id,
                    )

                else:

                    logger.info(
                        "✏️ [%s] No username replacement needed.",
                        msg.id,
                    )

                # ------------------------------------------------
                # Media.
                # ------------------------------------------------

                media = get_image_media(
                    msg
                )

                if media:

                    logger.info(
                        "🖼️ [%s] IMAGE/GIF DETECTED",
                        msg.id,
                    )

                    success = (
                        await process_single_media(
                            media=media,
                            caption_text=rewritten_text,
                            target_id=target_id,
                            message_id=msg.id,
                        )
                    )

                    if not success:

                        logger.error(
                            "❌ [%s] Image was NOT posted "
                            "because regeneration failed.",
                            msg.id,
                        )

                        # IMPORTANT:
                        #
                        # Do not advance last_processed
                        # on failure.
                        #
                        # This allows a later poll to retry
                        # the message.
                        #
                        continue

                else:

                    # ------------------------------------------------
                    # Text only.
                    # ------------------------------------------------

                    logger.info(
                        "📝 [%s] TEXT-ONLY MESSAGE",
                        msg.id,
                    )

                    if rewritten_text:

                        await bot.send_message(
                            chat_id=target_id,
                            text=rewritten_text,
                        )

                        logger.info(
                            "📝 [%s] Posted text-only message.",
                            msg.id,
                        )

                    else:

                        logger.info(
                            "📝 [%s] Empty text-only message. "
                            "Nothing posted.",
                            msg.id,
                        )

                # ------------------------------------------------
                # ONLY mark as processed AFTER SUCCESS.
                # ------------------------------------------------

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
                    "✅ [%s] Message successfully handled.",
                    msg.id,
                )

                logger.info(
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )

                # Small delay.
                await asyncio.sleep(
                    8
                )

            finally:

                async with processing_lock:

                    processing_ids.discard(
                        msg.id
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

        except Exception as restart_error:

            logger.exception(
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
    """
    Continuously poll registered source channels.
    """

    while True:

        try:

            if not await ensure_connection():

                await asyncio.sleep(
                    10
                )

                continue

            clients = (
                database.get_all_clients()
            )

            if not clients:

                logger.warning(
                    "⚠️ No clients registered in database."
                )

                await asyncio.sleep(
                    10
                )

                continue

            for client in clients:

                source_id = client[
                    "source"
                ]

                target_id = client[
                    "target"
                ]

                await process_channel(
                    source_id,
                    target_id,
                )

            await asyncio.sleep(
                5
            )

        except asyncio.CancelledError:

            raise

        except Exception:

            logger.exception(
                "❌ Polling loop error."
            )

            await asyncio.sleep(
                10
            )


# ============================================================
# INITIALIZE LAST PROCESSED
# ============================================================

async def initialize_last_processed():
    """
    Start from the newest message in each configured source.

    This prevents the first startup from regenerating all
    historical messages.
    """

    clients = (
        database.get_all_clients()
    )

    for client in clients:

        source = client[
            "source"
        ]

        try:

            channel = (
                await user_client.get_entity(
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
                    "📌 Source %s starts after message %s",
                    source,
                    msg.id,
                )

                break

        except Exception as e:

            logger.exception(
                "❌ Could not initialize source %s: %s",
                source,
                e,
            )


# ============================================================
# MAIN
# ============================================================

async def main():

    logger.info(
        "🚀 Starting Telegram Image Recreation Bot..."
    )

    logger.info(
        "📝 Caption handling: EXACT username replacement"
    )

    logger.info(
        "🖼️ Image handling: "
        "Vision → Deconstruction → Generation"
    )

    # --------------------------------------------------------
    # Start Telegram user client.
    # --------------------------------------------------------

    try:

        await user_client.start(
            phone=PHONE,
            force_sms=True,
        )

    except Exception:

        logger.exception(
            "❌ Failed to start Telegram client."
        )

        raise

    if not user_client.is_connected():

        logger.error(
            "❌ Telegram client is not connected!"
        )

        return

    logger.info(
        "✅ User client connected!"
    )

    # --------------------------------------------------------
    # Register default source → target.
    # --------------------------------------------------------

    try:

        existing = (
            database.get_target_for_source(
                SOURCE_CHANNEL_ID
            )
        )

        if existing is None:

            logger.info(
                "📝 Adding default client: "
                "%s → %s",
                SOURCE_CHANNEL_ID,
                TARGET_CHANNEL_ID,
            )

            database.add_client(
                SOURCE_CHANNEL_ID,
                TARGET_CHANNEL_ID,
            )

        else:

            logger.info(
                "✅ Default client already registered: "
                "%s → %s",
                SOURCE_CHANNEL_ID,
                existing,
            )

    except Exception:

        logger.exception(
            "❌ Failed to register default client."
        )

        raise

    # --------------------------------------------------------
    # Initialize processing position.
    # --------------------------------------------------------

    await initialize_last_processed()

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
