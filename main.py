import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from io import BytesIO

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from telethon import TelegramClient, errors
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
import database
from ai_processor import rewrite_text, regenerate_image_from_bytes, generate_image_from_description

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID")) if os.getenv("API_ID") else 0
API_HASH = os.getenv("API_HASH")
PHONE = os.getenv("PHONE_NUMBER")

if not BOT_TOKEN or not API_ID or not API_HASH or not PHONE:
    logger.error("Missing environment variables! Check .env file.")
    exit(1)

bot = Bot(token=BOT_TOKEN)
user_client = TelegramClient('session_name', API_ID, API_HASH)

SOURCE_CHANNEL_ID = -1003593544389
TARGET_CHANNEL_ID = -1004415621706

ENABLE_IMAGE_FOR_TEXT = os.getenv("ENABLE_IMAGE_FOR_TEXT", "true").lower() == "true"

last_processed = {}

# === NEW: Connection safety net ===
async def ensure_connection():
    """Check if the client is connected. If not, reconnect."""
    if not user_client.is_connected():
        logger.warning("⚠️ Client disconnected! Attempting to reconnect...")
        try:
            await user_client.connect()
            if not user_client.is_connected():
                logger.info("Reconnecting via .start()...")
                await user_client.start(phone=PHONE)
            logger.info("✅ Reconnected successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Reconnection failed: {e}")
            return False
    return True

async def process_channel(source_id, target_id):
    global last_processed
    
    # Step 0: Ensure we are connected BEFORE doing anything
    if not await ensure_connection():
        logger.error("❌ Cannot process: Client is disconnected and reconnection failed.")
        return

    try:
        channel = await user_client.get_entity(source_id)
        async for msg in user_client.iter_messages(channel, limit=1):
            if msg.id == last_processed.get(source_id):
                return

            last_processed[source_id] = msg.id
            logger.info(f"📩 New message in {source_id} (ID: {msg.id})")

            # --- Rewrite text ---
            original_text = msg.text or msg.caption or ""
            rewritten_text = await rewrite_text(original_text)

            # --- Check for media ---
            image_bytes = None
            media = msg.media
            final_image_url = None

            if media:
                logger.info(f"🔍 Media detected: {type(media)}")
                try:
                    image_bytes = await user_client.download_media(media, bytes)
                    if image_bytes:
                        logger.info("✅ Downloaded media bytes")
                    else:
                        logger.warning("Could not download media")
                except Exception as e:
                    logger.error(f"Error downloading media: {e}")

            # --- Case 1: We have media (image/GIF) -> Regenerate using Vision + DALL-E ---
            if image_bytes:
                logger.info("🔄 Regenerating image via Vision + DALL-E (removing watermarks)...")
                new_image_url = await regenerate_image_from_bytes(image_bytes)
                if new_image_url:
                    final_image_url = new_image_url
                    logger.info("✅ Successfully regenerated image")
                else:
                    logger.warning("Regeneration failed, falling back to original media")
                    await bot.send_photo(
                        chat_id=target_id,
                        photo=BytesIO(image_bytes),
                        caption=rewritten_text[:1024]
                    )
                    logger.info(f"📸 Posted original media (regeneration failed) to {target_id}")
                    return

            # --- Case 2: No media (text-only) -> Optionally generate image from text ---
            elif ENABLE_IMAGE_FOR_TEXT and rewritten_text and len(rewritten_text) > 10:
                logger.info("🖼️ No media, generating image from rewritten text (DALL-E)...")
                final_image_url = await generate_image_from_description(rewritten_text)
                if final_image_url:
                    logger.info("✅ Generated image from text")
                else:
                    logger.warning("Text-to-image generation failed, sending text only")

            # --- Post the result ---
            if final_image_url:
                await bot.send_photo(
                    chat_id=target_id,
                    photo=final_image_url,
                    caption=rewritten_text[:1024]
                )
                logger.info(f"📸 Posted with new image to {target_id}")
            else:
                await bot.send_message(chat_id=target_id, text=rewritten_text)
                logger.info(f"📝 Posted text-only to {target_id}")

            break
    except errors.rpcerrorlist.AuthKeyError as e:
        logger.error(f"Authentication error: {e}. Restarting client...")
        await user_client.start(phone=PHONE)
    except Exception as e:
        logger.error(f"Error processing {source_id}: {e}")

async def poll_channels():
    while True:
        # Ensure connection at the start of every loop iteration
        if not await ensure_connection():
            logger.warning("⚠️ Offline, waiting 10 seconds to retry...")
            await asyncio.sleep(10)
            continue

        clients = database.get_all_clients()
        if not clients:
            logger.warning("⚠️ No clients in database – waiting...")
            await asyncio.sleep(10)
            continue
        for client in clients:
            source = client["source"]
            target = client["target"]
            await process_channel(source, target)
        await asyncio.sleep(5)

async def main():
    logger.info("Starting Telegram Rewriter Bot (Hybrid Mode with Auto-Reconnect)...")
    
    # Initial connect
    await user_client.start(phone=PHONE)
    if not user_client.is_connected():
        await user_client.connect()
        if not user_client.is_connected():
            logger.error("❌ Failed to connect on startup!")
            return
    logger.info("User client connected!")

    existing = database.get_target_for_source(SOURCE_CHANNEL_ID)
    if existing is None:
        logger.info(f"📝 Adding client: {SOURCE_CHANNEL_ID} → {TARGET_CHANNEL_ID}")
        database.add_client(SOURCE_CHANNEL_ID, TARGET_CHANNEL_ID)
    else:
        logger.info(f"✅ Client already registered: {SOURCE_CHANNEL_ID} → {existing}")

    for client in database.get_all_clients():
        source = client["source"]
        try:
            channel = await user_client.get_entity(source)
            async for msg in user_client.iter_messages(channel, limit=1):
                last_processed[source] = msg.id
                logger.info(f"📌 Last message in {source}: {msg.id}")
        except Exception as e:
            logger.error(f"Could not fetch last message from {source}: {e}")

    logger.info("Starting polling loop...")
    await poll_channels()

if __name__ == "__main__":
    asyncio.run(main())
