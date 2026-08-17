import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# === LOAD ENV FIRST (before importing ai_processor) ===
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# === NOW IMPORT MODULES THAT USE ENV VARIABLES ===
from telethon import TelegramClient, events
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
import database
from ai_processor import rewrite_text, generate_image

# === SETUP LOGGING ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID")) if os.getenv("API_ID") else 0
API_HASH = os.getenv("API_HASH")
PHONE = os.getenv("PHONE_NUMBER")

if not BOT_TOKEN or not API_ID or not API_HASH or not PHONE:
    logger.error("Missing environment variables! Check .env file.")
    exit(1)

# Initialize the Bot (for posting to target channels)
bot = Bot(token=BOT_TOKEN)

# Initialize the User Client (for reading source channels without admin)
user_client = TelegramClient('session_name', API_ID, API_HASH)

# === EVENT HANDLER ===
@user_client.on(events.NewMessage)
async def handle_new_post(event):
    if not event.is_channel:
        return
    source_channel_id = event.chat_id
    target_channel_id = database.get_target_for_source(source_channel_id)
    if not target_channel_id:
        return
    logger.info(f"New post from source {source_channel_id} -> forwarding to {target_channel_id}")
    try:
        original_text = event.message.text or event.message.caption or ""
        rewritten_text = await rewrite_text(original_text)
        image_url = None
        if rewritten_text and len(rewritten_text) > 10:
            image_url = await generate_image(rewritten_text)
        if image_url:
            await bot.send_photo(
                chat_id=target_channel_id,
                photo=image_url,
                caption=rewritten_text[:1024]
            )
            logger.info(f"Posted with image to {target_channel_id}")
        else:
            await bot.send_message(
                chat_id=target_channel_id,
                text=rewritten_text
            )
            logger.info(f"Posted text-only to {target_channel_id}")
    except TelegramAPIError as e:
        logger.error(f"Failed to post to {target_channel_id}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

# === STARTUP ===
async def main():
    logger.info("Starting Telegram Rewriter Bot...")
    await user_client.start(phone=PHONE)
    logger.info("User client connected!")

    # ✅ To add a client, uncomment the line below, replace IDs, run locally once, then re‑comment.
    # database.add_client(source_channel_id=-1003593544389, target_channel_id=-1004415621706)

    logger.info("Bot is running. Listening for channel messages...")
    await user_client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())
