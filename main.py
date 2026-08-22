import os
import asyncio
import logging
from pathlib import Path

# ============================================================
# LOAD ENVIRONMENT FIRST
# ============================================================
# IMPORTANT:
# ai_processor.py reads environment variables when it is imported.
# Therefore .env MUST be loaded before importing ai_processor.

from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


# ============================================================
# IMPORTS
# ============================================================

from aiogram import Bot
from telethon import TelegramClient, errors

import database
from ai_processor import rewrite_text, regenerate_image_from_bytes


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID")) if os.getenv("API_ID") else 0
API_HASH = os.getenv("API_HASH")
PHONE = os.getenv("PHONE_NUMBER")
NEW_MENTION = os.getenv("NEW_MENTION", "").strip()


if NEW_MENTION and not NEW_MENTION.startswith("@"):
    NEW_MENTION = "@" + NEW_MENTION


if not BOT_TOKEN or not API_ID or not API_HASH or not PHONE:
    logger.error("Missing environment variables! Check your .env file.")
    raise SystemExit(1)


# ============================================================
# TELEGRAM CLIENTS
# ============================================================

bot = Bot(token=BOT_TOKEN)

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
# PROCESSING STATE
# ============================================================

# Prevent the same message from being processed twice
# during the lifetime of this process.
processing_ids = set()

processing_lock = asyncio.Lock()

# Last processed message per source channel.
last_processed = {}


# ============================================================
# CONNECTION
# ============================================================

async def ensure_connection():
    """
    Make sure the Telethon client is connected.
    """

    if user_client.is_connected():
        return True

    logger.warning("⚠️ Client disconnected! Attempting to reconnect...")

    try:
        await user_client.connect()

        if not user_client.is_connected():
            await user_client.start(phone=PHONE)

        logger.info("✅ Reconnected successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Reconnection failed: {e}")
        return False


# ============================================================
# MEDIA DETECTION
# ============================================================

def get_image_media(msg):
    """
    Return Telegram media if the message contains an image.

    Supports:
    - Telegram photos
    - GIFs
    - animated GIFs sent as documents
    - image documents
    """

    # Standard Telegram photo
    if getattr(msg, "photo", None):
        return msg.photo

    # Telegram GIFs/images are commonly documents.
    document = getattr(msg, "document", None)

    if document:
        mime_type = getattr(document, "mime_type", "") or ""

        if mime_type.startswith("image/"):
            return document

        # Some Telegram GIFs can be reported differently.
        file_name = getattr(document, "attributes", None)

        if file_name:
            return document

    # Extra Telethon fallback
    media = getattr(msg, "media", None)

    if media and hasattr(media, "photo"):
        return media

    return None


# ============================================================
# IMAGE PROCESSING
# ============================================================

async def process_single_media(
    media,
    caption_text,
    target_id,
    message_id,
):
    """
    Download the original image/GIF and send it through:

        ORIGINAL IMAGE/GIF
              ↓
        OpenAI Vision
              ↓
        Visual deconstruction
              ↓
        OpenAI image generation
              ↓
        NEW IMAGE
              ↓
        Telegram target

    The original image is NEVER reposted if generation fails.
    """

    try:
        logger.info(
            f"🖼️ [{message_id}] Downloading original image/GIF..."
        )

        image_bytes = await user_client.download_media(
            media,
            bytes,
        )

        if not image_bytes:
            logger.error(
                f"❌ [{message_id}] Could not download image."
            )
            return False

        logger.info(
            f"📦 [{message_id}] Downloaded "
            f"{len(image_bytes)} bytes"
        )

        logger.info(
            f"🔍 [{message_id}] Calling image regeneration pipeline..."
        )

        try:
            new_image_data = await regenerate_image_from_bytes(
                image_bytes
            )
        except Exception:
            logger.exception(
                f"❌ [{message_id}] UNHANDLED REGENERATION EXCEPTION"
            )
            return False

        if not new_image_data:
            logger.error(
                f"❌ [{message_id}] Image regeneration failed. "
                f"Original image will NOT be reposted."
            )
            return False

        logger.info(
            f"📤 [{message_id}] Sending NEW regenerated image..."
        )

        await bot.send_photo(
            chat_id=target_id,
            photo=new_image_data,
            caption=caption_text[:1024] if caption_text else None,
        )

        logger.info(
            f"✅ [{message_id}] NEW regenerated image posted."
        )

        return True

    except Exception as e:
        logger.exception(
            f"❌ [{message_id}] Error processing image: {e}"
        )
        return False


# ============================================================
# CHANNEL PROCESSING
# ============================================================

async def process_channel(source_id, target_id):
    """
    Process new messages from one source channel.
    """

    if not await ensure_connection():
        logger.error(
            "❌ Cannot process channel: client is disconnected."
        )
        return

    try:
        channel = await user_client.get_entity(source_id)

        last_id = last_processed.get(
            source_id,
            0,
        )

        # --------------------------------------------------------
        # Fetch new messages in chronological order.
        # --------------------------------------------------------

        new_messages = []

        async for msg in user_client.iter_messages(
            channel,
            min_id=last_id,
            reverse=True,
        ):
            new_messages.append(msg)

        if not new_messages:
            return

        # --------------------------------------------------------
        # Process each message.
        # --------------------------------------------------------

        for msg in new_messages:

            async with processing_lock:

                if msg.id in processing_ids:
                    logger.warning(
                        f"⚠️ Message {msg.id} is already being processed. "
                        f"Skipping duplicate."
                    )
                    continue

                processing_ids.add(msg.id)

            try:

                logger.info(
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )

                logger.info(
                    f"📩 Processing message ID: {msg.id}"
                )

                # ------------------------------------------------
                # Extract caption/text.
                # ------------------------------------------------

                original_text = (
                    getattr(msg, "text", None)
                    or getattr(msg, "message", None)
                    or getattr(msg, "caption", None)
                    or ""
                )

                # ------------------------------------------------
                # ONLY replace username.
                #
                # No DeepSeek.
                # No AI rewriting.
                # No paraphrasing.
                # ------------------------------------------------

                rewritten_text = await rewrite_text(
                    original_text
                )

                if original_text != rewritten_text:
                    logger.info(
                        f"✏️ [{msg.id}] Username replacement applied."
                    )
                else:
                    logger.info(
                        f"✏️ [{msg.id}] No username replacement needed."
                    )

                # ------------------------------------------------
                # Detect image/GIF.
                # ------------------------------------------------

                media = get_image_media(msg)

                if media:

                    logger.info(
                        f"🖼️ [{msg.id}] IMAGE/GIF DETECTED"
                    )

                    logger.info(
                        f"🔍 [{msg.id}] Starting:"
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

                    success = await process_single_media(
                        media=media,
                        caption_text=rewritten_text,
                        target_id=target_id,
                        message_id=msg.id,
                    )

                    if not success:
                        logger.error(
                            f"❌ [{msg.id}] Image was NOT posted "
                            f"because regeneration failed."
                        )

                else:

                    # ------------------------------------------------
                    # TEXT ONLY MESSAGE
                    # ------------------------------------------------

                    logger.info(
                        f"📝 [{msg.id}] TEXT-ONLY MESSAGE"
                    )

                    if rewritten_text:

                        await bot.send_message(
                            chat_id=target_id,
                            text=rewritten_text,
                        )

                        logger.info(
                            f"📝 [{msg.id}] Posted text-only message."
                        )

                # ------------------------------------------------
                # Mark message as handled.
                # ------------------------------------------------

                last_processed[source_id] = max(
                    last_processed.get(source_id, 0),
                    msg.id,
                )

                logger.info(
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )

                # Small delay between messages.
                await asyncio.sleep(8)

            finally:

                async with processing_lock:
                    processing_ids.discard(msg.id)

    except errors.rpcerrorlist.AuthKeyError as e:

        logger.error(
            f"Authentication error: {e}. Restarting..."
        )

        try:
            await user_client.start(
                phone=PHONE
            )

        except Exception as restart_error:

            logger.error(
                f"❌ Could not restart Telegram client: "
                f"{restart_error}"
            )

    except Exception as e:

        logger.exception(
            f"❌ Error processing channel "
            f"{source_id}: {e}"
        )


# ============================================================
# POLLING
# ============================================================

async def poll_channels():

    while True:

        if not await ensure_connection():
            await asyncio.sleep(10)
            continue

        clients = database.get_all_clients()

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
        "🚀 Starting Telegram Image Recreation Bot..."
    )

    logger.info(
        "   Caption handling: remove '*' + exact username replacement"
    )

    logger.info(
        "   NEW_MENTION: %r",
        NEW_MENTION,
    )

    logger.info(
        "   Image handling: Vision → Deconstruction → Generation"
    )

    # --------------------------------------------------------
    # Start Telegram user client.
    # --------------------------------------------------------

    await user_client.start(
        phone=PHONE,
        force_sms=True,
    )

    if not user_client.is_connected():

        await user_client.connect()

        if not user_client.is_connected():

            logger.error(
                "❌ Failed to connect on startup!"
            )

            return

    logger.info(
        "✅ User client connected!"
    )

    # --------------------------------------------------------
    # Register source → target.
    # --------------------------------------------------------

    existing = database.get_target_for_source(
        SOURCE_CHANNEL_ID
    )

    if existing is None:

        logger.info(
            f"📝 Adding client: "
            f"{SOURCE_CHANNEL_ID} → {TARGET_CHANNEL_ID}"
        )

        database.add_client(
            SOURCE_CHANNEL_ID,
            TARGET_CHANNEL_ID,
        )

    else:

        logger.info(
            f"✅ Client already registered: "
            f"{SOURCE_CHANNEL_ID} → {existing}"
        )

    # --------------------------------------------------------
    # Start from newest message.
    #
    # This prevents the bot from regenerating every old post
    # when it starts for the first time.
    # --------------------------------------------------------

    for client in database.get_all_clients():

        source = client["source"]

        try:

            channel = await user_client.get_entity(
                source
            )

            async for msg in user_client.iter_messages(
                channel,
                limit=1,
            ):

                last_processed[source] = msg.id

                logger.info(
                    f"📌 Last message in {source}: "
                    f"{msg.id}"
                )

        except Exception as e:

            logger.error(
                f"Could not fetch last message "
                f"from {source}: {e}"
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
    asyncio.run(main())
