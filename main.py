import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

from telethon import TelegramClient
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
import database
from ai_processor import rewrite_text, regenerate_image_from_url

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
    global last_processed
    try:
        channel = await user_client.get_entity(source_id)
        async for msg in user_client.iter_messages(channel, limit=1):
            if msg.id == last_processed.get(source_id):
                return
            logger.info(f"📩 New message in {source_id} (ID: {msg.id})")

            # --- Extract text ---
            original_text = msg.text or msg.caption or ""
            rewritten_text = await rewrite_text(original_text)

            # --- Check for media (photo or GIF) ---
            image_url = None
            if msg.photo:
                # Get the largest photo size
                photo = msg.photo
                file = await user_client.download_media(photo, bytes)
                if file:
                    # We need a URL; we'll use the file ID to get a direct link
                    # Telethon doesn't provide a direct URL; we can upload to a hosting or use the file_id with bot API
                    # Simplest: we use the bot API to get the file path and then construct a URL
                    # But easier: use the bot to get the file and send it
                    # However, we can just use the built-in `get_file_url` trick:
                    # For simplicity, we'll use the message's media group ID or download and re-upload? 
                    # Actually we can use the bot's getFile method to get a URL.
                    # But to avoid complexity, we can skip Vision for now and just send the image as is.
                    # For Option 3, we need an accessible URL. We'll use a free image host or use the bot's file URL.
                    # Let's implement a helper to get a public URL for the photo via the bot API.
                    file_info = await bot.get_file(msg.photo.file_id)
                    file_path = file_info.file_path
                    image_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
                else:
                    logger.warning("Could not download photo")
            elif msg.document and msg.document.mime_type and 'gif' in msg.document.mime_type:
                # It's a GIF
                file = await user_client.download_media(msg.document, bytes)
                if file:
                    # We need a URL; we'll use the file ID similarly
                    file_info = await bot.get_file(msg.document.file_id)
                    file_path = file_info.file_path
                    image_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
                else:
                    logger.warning("Could not download GIF")

            # If we have an image URL, regenerate it
            if image_url:
                new_image_url = await regenerate_image_from_url(image_url)
                if new_image_url:
                    await bot.send_photo(
                        chat_id=target_id,
                        photo=new_image_url,
                        caption=rewritten_text[:1024]
                    )
                    logger.info(f"📸 Posted regenerated image to {target_id}")
                else:
                    # Fallback: send original image with rewritten text
                    await bot.send_photo(
                        chat_id=target_id,
                        photo=image_url,
                        caption=rewritten_text[:1024]
                    )
                    logger.info(f"📸 Posted original image (regeneration failed) to {target_id}")
            else:
                # Text-only
                await bot.send_message(chat_id=target_id, text=rewritten_text)
                logger.info(f"📝 Posted text-only to {target_id}")

            last_processed[source_id] = msg.id
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
