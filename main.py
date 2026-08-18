import os
import asyncio
import logging
import re
from pathlib import Path
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
import database
from ai_processor import rewrite_text, generate_image
import io
from PIL import Image

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID")) if os.getenv("API_ID") else 0
API_HASH = os.getenv("API_HASH")
PHONE = os.getenv("PHONE_NUMBER")

if not BOT_TOKEN or not API_ID or not API_HASH or not PHONE:
    logger.error("Missing environment variables!")
    exit(1)

bot = Bot(token=BOT_TOKEN)
user_client = TelegramClient('session_name', API_ID, API_HASH)

SOURCE_CHANNEL_ID = -1003593544389
TARGET_CHANNEL_ID = -1004415621706

last_processed = {}

def extract_first_frame_from_gif(data: bytes) -> bytes:
    """Extract the first frame of a GIF and return as JPEG bytes."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.is_animated:
                img.seek(0)  # first frame
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=85)
            return output.getvalue()
    except Exception as e:
        logger.error(f"GIF extraction failed: {e}")
        return None

async def process_channel(source_id, target_id):
    global last_processed
    try:
        channel = await user_client.get_entity(source_id)
        async for msg in user_client.iter_messages(channel, limit=1):
            if msg.id == last_processed.get(source_id):
                return
            logger.info(f"📩 New message in {source_id} (ID: {msg.id})")

            original_text = msg.text or msg.caption or ""
            rewritten_text = await rewrite_text(original_text)

            image_bytes = None
            # Check if message has media
            if msg.media:
                # Download the media
                media_data = await user_client.download_file(msg.media, bytes)
                if media_data:
                    # Determine if it's an image or GIF
                    mime_type = getattr(msg.media, 'mime_type', '')
                    if 'image/gif' in mime_type:
                        # GIF – extract first frame
                        image_bytes = extract_first_frame_from_gif(media_data)
                    elif 'image' in mime_type:
                        image_bytes = media_data  # keep as is (will be sent as photo)
                    # else: ignore other media types

            # If we have an image (from photo or GIF), we can generate a new image using DALL-E
            # based on the rewritten text, which removes any watermark.
            # Alternatively, we could send the original image as is, but we want to remove watermark.
            # We'll generate a new image using DALL-E.
            image_url = None
            if rewritten_text and len(rewritten_text) > 10:
                image_url = await generate_image(rewritten_text)

            # Post to target
            if image_url:
                await bot.send_photo(chat_id=target_id, photo=image_url, caption=rewritten_text[:1024])
                logger.info(f"✅ Posted with generated image to {target_id}")
            else:
                # fallback: send just text
                await bot.send_message(chat_id=target_id, text=rewritten_text)
                logger.info(f"✅ Posted text-only to {target_id}")

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
    logger.info("Starting Telegram Rewriter Bot (Polling Mode)...")
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
