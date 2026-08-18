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
from ai_processor import rewrite_text, generate_image  # both functions

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

# ===== YOUR CHANNEL IDs (already set) =====
SOURCE_CHANNEL_ID = -1003593544389   # CAPPERS FREE🚦
TARGET_CHANNEL_ID = -1004415621706   # Caps_picks

last_processed = {}

async def process_channel(source_id, target_id):
    global last_processed
    try:
        channel = await user_client.get_entity(source_id)
        async for msg in user_client.iter_messages(channel, limit=1):
            # Skip if already processed
            if msg.id == last_processed.get(source_id):
                return

            logger.info(f"📩 New message in {source_id} (ID: {msg.id})")
            original_text = msg.text or msg.caption or ""

            # 1. Rewrite text
            rewritten_text = await rewrite_text(original_text)

            # 2. Generate image (if there is text to base prompt on)
            image_url = None
            if rewritten_text and len(rewritten_text) > 10:
                image_url = await generate_image(rewritten_text)

            # 3. Send to target channel
            if image_url:
                await bot.send_photo(
                    chat_id=target_id,
                    photo=image_url,
                    caption=rewritten_text[:1024]  # Telegram caption limit
                )
                logger.info(f"📸 Posted with image to {target_id}")
            else:
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
        await asyncio.sleep(5)  # check every 5 seconds

async def main():
    logger.info("Starting Telegram Rewriter Bot (with Image Generation)...")
    await user_client.start(phone=PHONE)
    logger.info("User client connected!")

    # Auto‑register your client if not already in DB
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
