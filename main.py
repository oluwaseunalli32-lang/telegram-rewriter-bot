import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from io import BytesIO
from aiogram.types import BufferedInputFile

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

async def ensure_connection():
    if not user_client.is_connected():
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
    return True

async def process_single_media(media, caption_text, target_id):
    """Process a single media item – regenerate or fallback."""
    try:
        image_bytes = await user_client.download_media(media, bytes)
        if not image_bytes:
            logger.warning("Could not download media")
            return

        logger.info("🔄 Regenerating image via Vision + DALL-E...")
        new_image_data = await regenerate_image_from_bytes(image_bytes)
        
        if new_image_data:
            # new_image_data is either URL (str) or BufferedInputFile
            await bot.send_photo(
                chat_id=target_id,
                photo=new_image_data,
                caption=caption_text[:1024] if caption_text else None
            )
            logger.info("📸 Posted regenerated image")
        else:
            logger.warning("Regeneration failed, falling back to original media")
            input_file = BufferedInputFile(file=image_bytes, filename="original_image.jpg")
            await bot.send_photo(
                chat_id=target_id,
                photo=input_file,
                caption=caption_text[:1024] if caption_text else None
            )
            logger.info("📸 Posted original media (regeneration failed)")
    except Exception as e:
        logger.error(f"Error processing media: {e}")

async def process_channel(source_id, target_id):
    global last_processed
    
    if not await ensure_connection():
        logger.error("❌ Cannot process: Client is disconnected.")
        return

    try:
        channel = await user_client.get_entity(source_id)
        last_id = last_processed.get(source_id, 0)
        
        new_messages = []
        async for msg in user_client.iter_messages(channel, min_id=last_id, reverse=True):
            new_messages.append(msg)
        
        if not new_messages:
            return

        last_processed[source_id] = new_messages[-1].id

        for msg in new_messages:
            logger.info(f"📩 Processing message ID: {msg.id}")

            original_text = msg.text or getattr(msg, 'caption', '') or ""
            rewritten_text = await rewrite_text(original_text)

            media_list = []
            if msg.photo:
                media_list.append(msg.photo)
            elif msg.document and msg.document.mime_type and ('image' in msg.document.mime_type or 'gif' in msg.document.mime_type):
                media_list.append(msg.document)
            elif msg.media and hasattr(msg.media, 'photo'):
                media_list.append(msg.media)

            if media_list:
                logger.info(f"📸 Found {len(media_list)} media items")
                for idx, media in enumerate(media_list):
                    caption = rewritten_text[:1024] if idx == 0 else None
                    await process_single_media(media, caption, target_id)
                    if idx < len(media_list) - 1:
                        await asyncio.sleep(4)
            else:
                if ENABLE_IMAGE_FOR_TEXT and rewritten_text and len(rewritten_text) > 10:
                    logger.info("🖼️ Generating image from text...")
                    image_data = await generate_image_from_description(rewritten_text)
                    if image_data:
                        await bot.send_photo(
                            chat_id=target_id,
                            photo=image_data,
                            caption=rewritten_text[:1024]
                        )
                        logger.info("📸 Posted generated image from text")
                    else:
                        await bot.send_message(chat_id=target_id, text=rewritten_text)
                        logger.info("📝 Posted text-only (image gen failed)")
                else:
                    await bot.send_message(chat_id=target_id, text=rewritten_text)
                    logger.info("📝 Posted text-only")

            await asyncio.sleep(8)

    except errors.rpcerrorlist.AuthKeyError as e:
        logger.error(f"Authentication error: {e}. Restarting...")
        await user_client.start(phone=PHONE)
    except Exception as e:
        logger.error(f"Error processing {source_id}: {e}")

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
            await process_channel(client["source"], client["target"])
        await asyncio.sleep(5)

async def main():
    logger.info("Starting Telegram Rewriter Bot (Multi‑Message + Retry + Watermark Removal)...")
    
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
