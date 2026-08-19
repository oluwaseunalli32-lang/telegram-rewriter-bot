import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from io import BytesIO

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from telethon import TelegramClient
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
import database
from ai_processor import rewrite_text, regenerate_image_from_bytes

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

last_processed = {}

async def process_channel(source_id, target_id):
    """
    Fetch the latest message from the source channel.
    If it's new, process it (rewrite + regenerate image) and post to target.
    """
    try:
        channel = await user_client.get_entity(source_id)
        async for msg in user_client.iter_messages(channel, limit=1):
            # If we've already processed this message ID, skip
            if msg.id == last_processed.get(source_id):
                return

            # Mark as processed immediately (prevents infinite loops)
            last_processed[source_id] = msg.id

            logger.info(f"📩 New message in {source_id} (ID: {msg.id})")

            # 1. Rewrite text
            original_text = msg.text or msg.caption or ""
            rewritten_text = await rewrite_text(original_text)

            # 2. Check for media (photo or GIF)
            image_bytes = None
            if msg.photo:
                try:
                    image_bytes = await user_client.download_media(msg.photo, bytes)
                    if image_bytes:
                        logger.info("✅ Downloaded photo bytes")
                    else:
                        logger.warning("Could not download photo")
                except Exception as e:
                    logger.error(f"Error downloading photo: {e}")

            elif msg.document and msg.document.mime_type and 'gif' in msg.document.mime_type:
                try:
                    image_bytes = await user_client.download_media(msg.document, bytes)
                    if image_bytes:
                        logger.info("✅ Downloaded GIF bytes")
                    else:
                        logger.warning("Could not download GIF")
                except Exception as e:
                    logger.error(f"Error downloading GIF: {e}")

            # 3. If we have image bytes, try to regenerate
            if image_bytes:
                new_image_url = await regenerate_image_from_bytes(image_bytes)
                if new_image_url:
                    await bot.send_photo(
                        chat_id=target_id,
                        photo=new_image_url,
                        caption=rewritten_text[:1024]
                    )
                    logger.info(f"📸 Posted regenerated image to {target_id}")
                else:
                    # Fallback: send original image (as bytes)
                    await bot.send_photo(
                        chat_id=target_id,
                        photo=BytesIO(image_bytes),
                        caption=rewritten_text[:1024]
                    )
                    logger.info(f"📸 Posted original image (regeneration failed) to {target_id}")
            else:
                # Text-only
                await bot.send_message(chat_id=target_id, text=rewritten_text)
                logger.info(f"📝 Posted text-only to {target_id}")

            # Exit the loop after processing the latest message
            break
    except Exception as e:
        logger.error(f"Error processing {source_id}: {e}")

async def poll_channels():
    while True:
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
    logger.info("Starting Telegram Rewriter Bot (Vision + Regeneration Mode)...")
    await user_client.start(phone=PHONE)
    logger.info("User client connected!")

    # Auto-register the client if not present
    existing = database.get_target_for_source(SOURCE_CHANNEL_ID)
    if existing is None:
        logger.info(f"📝 Adding client: {SOURCE_CHANNEL_ID} → {TARGET_CHANNEL_ID}")
        database.add_client(SOURCE_CHANNEL_ID, TARGET_CHANNEL_ID)
    else:
        logger.info(f"✅ Client already registered: {SOURCE_CHANNEL_ID} → {existing}")

    # Initialize last_processed for all source channels
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
