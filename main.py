import os
import asyncio
import logging
from dotenv import load_dotenv
from telethon import TelegramClient, events
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError

import database
from ai_processor import rewrite_text, generate_image

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
PHONE = os.getenv("PHONE_NUMBER")

# Initialize the Bot (for posting to target channels)
bot = Bot(token=BOT_TOKEN)

# Initialize the User Client (for reading source channels without admin)
user_client = TelegramClient('session_name', API_ID, API_HASH)

# === EVENT HANDLER: Detects new posts in ANY channel the user is in ===
@user_client.on(events.NewMessage)
async def handle_new_post(event):
    # Ignore private chats, only process channel messages
    if not event.is_channel:
        return
    
    source_channel_id = event.chat_id
    
    # Check if this source channel is registered in our database
    target_channel_id = database.get_target_for_source(source_channel_id)
    if not target_channel_id:
        return  # Not a client's source channel, ignore
    
    logger.info(f"New post from source {source_channel_id} -> forwarding to {target_channel_id}")
    
    try:
        # Extract text
        original_text = event.message.text or event.message.caption or ""
        media = event.message.media
        
        # Step 1: Rewrite the text using DeepSeek
        rewritten_text = await rewrite_text(original_text)
        
        # Step 2: Generate an image using OpenAI DALL-E (if text exists)
        image_url = None
        if rewritten_text and len(rewritten_text) > 10:
            image_url = await generate_image(rewritten_text)
        
        # Step 3: Post to target channel using the Bot (must be admin there)
        if image_url:
            await bot.send_photo(
                chat_id=target_channel_id,
                photo=image_url,
                caption=rewritten_text[:1024]  # Telegram caption limit
            )
            logger.info(f"Posted with image to {target_channel_id}")
        else:
            # If no image, just send text
            await bot.send_message(
                chat_id=target_channel_id,
                text=rewritten_text
            )
            logger.info(f"Posted text-only to {target_channel_id}")
            
    except TelegramAPIError as e:
        logger.error(f"Failed to post to {target_channel_id}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error processing post from {source_channel_id}: {e}")

# === STARTUP ===
async def main():
    logger.info("Starting Telegram Rewriter Bot...")
    
    # Start the user client (this will prompt for phone code on first run)
    await user_client.start(phone=PHONE)
    logger.info("User client connected!")
    
    # (Optional) Add a test client entry to the DB – remove this later!
    # You can uncomment and run once to register your test channels.
    # database.add_client(source_channel_id=-1001234567890, target_channel_id=-1009876543210)
    
    logger.info("Bot is running. Listening for channel messages...")
    await user_client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())
