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
from ai_processor import rewrite_text, generate_image

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

# Store last processed message ID for each source channel
last_processed = {}

async def process_channel(source_id, target_id):
    """Check for new messages in the source channel and process them."""
    global last_processed
    try:
        # Get the channel entity
        channel = await user_client.get_entity(source_id)
        # Get the latest message
        async for msg in user_client.iter_messages(channel, limit=1):
            if msg.id == last_processed.get(source_id):
                return  # No new message
            # Process the new message
            logger.info(f"📩 New message in {source_id} (ID: {msg.id})")
            original_text = msg.text or msg.caption or ""
            rewritten_text = await rewrite_text(original_text)
            image_url = None
            if rewritten_text and len(rewritten_text) > 10:
                image_url = await generate_image(rewritten_text)
            if image_url:
                await bot.send_photo(chat_id=target_id, photo=image_url, caption=rewritten_text[:1024])
            else:
                await bot.send_message(chat_id=target_id, text=rewritten_text)
            last_processed[source_id] = msg.id
            logger.info(f"✅ Posted to {target_id}")
            break
    except Exception as e:
        logger.error(f"Error processing {source_id}: {e}")

async def poll_channels():
    """Continuously poll all registered source channels."""
    while True:
        clients = database.get_all_clients()
        for client in clients:
            source = client["source"]
            target = client["target"]
            await process_channel(source, target)
        await asyncio.sleep(5)  # Check every 5 seconds

async def main():
    logger.info("Starting Telegram Rewriter Bot (Polling Mode)...")
    await user_client.start(phone=PHONE)
    logger.info("User client connected!")

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
