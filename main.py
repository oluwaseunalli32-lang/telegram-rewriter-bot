import os
import asyncio
import logging
from pathlib import Path

# ============================================================
# LOAD ENVIRONMENT FIRST
# ============================================================

from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"

load_dotenv(
    dotenv_path=env_path
)


# ============================================================
# IMPORTS
# ============================================================

from aiogram import Bot
from telethon import TelegramClient, errors

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

logger = logging.getLogger(__name__)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

BOT_TOKEN = (
    os.getenv(
        "BOT_TOKEN"
    ) or ""
).strip()

API_ID_RAW = (
    os.getenv(
        "API_ID"
    ) or ""
).strip()

API_HASH = (
    os.getenv(
        "API_HASH"
    ) or ""
).strip()

PHONE = (
    os.getenv(
        "PHONE_NUMBER"
    ) or ""
).strip()


try:

    API_ID = int(
        API_ID_RAW
    )

except ValueError:

    API_ID = 0


# ============================================================
# VALIDATE CONFIG
# ============================================================

if not BOT_TOKEN:

    logger.error(
        "❌ BOT_TOKEN is missing."
    )

if not API_ID:

    logger.error(
        "❌ API_ID is missing or invalid."
    )

if not API_HASH:

    logger.error(
        "❌ API_HASH is missing."
    )

if not PHONE:

    logger.error(
        "❌ PHONE_NUMBER is missing."
    )

if (
    not BOT_TOKEN
    or not API_ID
    or not API_HASH
    or not PHONE
):

    raise SystemExit(
        1
    )


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
        "⚠️ Client disconnected! "
        "Attempting to reconnect..."
    )

    try:

        await user_client.connect()

        if not user_client.is_connected():

            await user_client.start(
                phone=PHONE
            )

        if user_client.is_connected():

            logger.info(
                "✅ Reconnected successfully!"
            )

            return True

        logger.error(
            "❌ Reconnection did not establish connection."
        )

        return False

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
    Return Telegram media that can be processed.

    Supports:

    - Telegram photos
    - image documents
    - GIFs
    - MP4 GIFs
    - videos
    - MOV
    - WebM
    """

    # --------------------------------------------------------
    # TELEGRAM PHOTO
    # --------------------------------------------------------

    if getattr(
        msg,
        "photo",
        None,
    ):

        return msg.photo

    # --------------------------------------------------------
    # DOCUMENT
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

        # Images
        if mime_type.startswith(
            "image/"
        ):

            return document

        # Videos / Telegram GIFs
        if mime_type.startswith(
            "video/"
        ):

            return document

        # File extension fallback
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

            if extension in supported_extensions:

                return document

    # --------------------------------------------------------
    # FALLBACK MEDIA
    # --------------------------------------------------------

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
# PROCESS SINGLE IMAGE
# ============================================================

async def process_single_media(
    media,
    caption_text,
    target_id,
    message_id,
):
    """
    Download original media.

    Then:

        ORIGINAL
            ↓
        VISION
            ↓
        DECONSTRUCTION
            ↓
        GENERATION
            ↓
        NEW IMAGE
            ↓
        TARGET

    Original media is never posted if regeneration fails.
    """

    try:

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
                "❌ [%s] Could not download image.",
                message_id,
            )

            return False

        logger.info(
            "📦 [%s] Downloaded %d bytes",
            message_id,
            len(image_bytes),
        )

        logger.info(
            "🔬 [%s] Media header: %s",
            message_id,
            bytes(
                image_bytes[:32]
            ).hex(" "),
        )

        # ----------------------------------------------------
        # REGENERATE
        # ----------------------------------------------------

        logger.info(
            "🔍 [%s] Sending original media "
            "through OpenAI pipeline...",
            message_id,
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
        # POST NEW IMAGE
        # ----------------------------------------------------

        logger.info(
            "📤 [%s] Sending NEW regenerated image...",
            message_id,
        )

        final_caption = (
            caption_text[:1024]
            if caption_text
            else None
        )

        await bot.send_photo(
            chat_id=target_id,
            photo=new_image_data,
            caption=final_caption,
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
            "❌ Cannot process channel: "
            "client is disconnected."
        )

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

        logger.info(
            "📥 Found %d new message(s) "
            "in source %s",
            len(new_messages),
            source_id,
        )

        # ----------------------------------------------------
        # PROCESS MESSAGES
        # ----------------------------------------------------

        for msg in new_messages:

            async with processing_lock:

                if msg.id in processing_ids:

                    logger.warning(
                        "⚠️ Message %s is already "
                        "being processed. Skipping.",
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
                # CAPTION / TEXT
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
                # MEDIA
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

                else:

                    # ------------------------------------------------
                    # TEXT ONLY
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

                # ------------------------------------------------
                # MARK PROCESSED
                # ------------------------------------------------

                last_processed[source_id] = max(
                    last_processed.get(
                        source_id,
                        0,
                    ),
                    msg.id,
                )

                logger.info(
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )

                # Small delay between messages.
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
            "❌ Error processing channel "
            "%s: %s",
            source_id,
            e,
        )


# ============================================================
# POLLING
# ============================================================

async def poll_channels():

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
                    "⚠️ No source/target clients configured."
                )

                await asyncio.sleep(
                    10
                )

                continue

            for client in clients:

                source = client["source"]
                target = client["target"]

                await process_channel(
                    source,
                    target,
                )

            await asyncio.sleep(
                5
            )

        except Exception as e:

            logger.exception(
                "❌ Polling loop error: %s",
                e,
            )

            await asyncio.sleep(
                10
            )


# ============================================================
# MAIN
# ============================================================

async def main():

    logger.info(
        "🚀 Starting Telegram Image Recreation Bot..."
    )

    logger.info(
        "   Caption handling: EXACT username replacement"
    )

    logger.info(
        "   Image handling: Vision → Deconstruction → Generation"
    )

    # --------------------------------------------------------
    # START TELEGRAM
    # --------------------------------------------------------

    await user_client.start(
        phone=PHONE,
        force_sms=True,
    )

    if not user_client.is_connected():

        logger.error(
            "❌ Failed to connect on startup!"
        )

        return

    logger.info(
        "✅ User client connected!"
    )

    # --------------------------------------------------------
    # REGISTER DEFAULT SOURCE → TARGET
    # --------------------------------------------------------

    source_channel_id = -1003593544389
    target_channel_id = -1004415621706

    existing = (
        database.get_target_for_source(
            source_channel_id
        )
    )

    if existing is None:

        logger.info(
            "📝 Adding client: %s → %s",
            source_channel_id,
            target_channel_id,
        )

        database.add_client(
            source_channel_id,
            target_channel_id,
        )

    else:

        logger.info(
            "✅ Client already registered: "
            "%s → %s",
            source_channel_id,
            existing,
        )

    # --------------------------------------------------------
    # START FROM NEWEST MESSAGE
    # --------------------------------------------------------

    for client in database.get_all_clients():

        source = client["source"]

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

                last_processed[source] = (
                    msg.id
                )

                logger.info(
                    "📌 Last message in %s: %s",
                    source,
                    msg.id,
                )

                break

        except Exception as e:

            logger.exception(
                "❌ Could not fetch last message "
                "from %s: %s",
                source,
                e,
            )

    # --------------------------------------------------------
    # POLLING
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
