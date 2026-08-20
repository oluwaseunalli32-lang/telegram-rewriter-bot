import os
import re
import base64
import time
import asyncio
from io import BytesIO
from PIL import Image
from openai import OpenAI
import replicate

# --- Clients ---
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# --- Config ---
NEW_MENTION = os.getenv("NEW_MENTION", "")
OLD_MENTION = "@cappersfree"

# 速率限制保护
_last_vision_call = 0
_vision_cooldown = 8  # 两次 Vision 调用间隔（秒）
_last_generation_call = 0
_generation_cooldown = 8  # 两次图像生成调用间隔（秒）

def clean_mentions(text: str) -> str:
    if not text:
        return text
    if NEW_MENTION:
        text = re.sub(OLD_MENTION, NEW_MENTION, text, flags=re.IGNORECASE)
    else:
        text = re.sub(r'@\w+', '', text)
    return text

def clean_text(text: str) -> str:
    if not text:
        return text
    text = clean_mentions(text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def rewrite_text(original_text: str) -> str:
    cleaned = clean_text(original_text)
    if not cleaned or len(cleaned) < 5:
        return original_text
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Rewrite the following text to make it unique while preserving the core meaning. Return ONLY the rewritten text, nothing else. Do not add prefixes like 'Here is the rewritten text' or 'Rewritten version:'. Just output the rewritten content."},
                {"role": "user", "content": cleaned}
            ],
            temperature=0.8,
            max_tokens=2000
        )
        rewritten = response.choices[0].message.content.strip()
        for prefix in ["Here is the rewritten text:", "Rewritten version:", "Here is:", "Rewritten:"]:
            if rewritten.lower().startswith(prefix.lower()):
                rewritten = rewritten[len(prefix):].strip()
        rewritten = clean_mentions(rewritten)
        rewritten = re.sub(r'\*\*([^*]+)\*\*', r'\1', rewritten)
        return rewritten
    except Exception as e:
        print(f"DeepSeek error: {e}")
        return original_text

async def describe_image_bytes(image_bytes: bytes) -> str:
    """使用 gpt-4o-mini 进行图像描述，带速率限制和重试。"""
    global _last_vision_call

    # 强制冷却
    now = time.time()
    elapsed = now - _last_vision_call
    if elapsed < _vision_cooldown:
        wait = _vision_cooldown - elapsed
        print(f"⏳ 等待 {wait:.1f}s 后进行下一次 Vision 调用...")
        await asyncio.sleep(wait)
    _last_vision_call = time.time()

    try:
        img = Image.open(BytesIO(image_bytes))
        if getattr(img, 'is_animated', False):
            img.seek(0)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        data_url = f"data:image/jpeg;base64,{img_base64}"

        vision_prompt = """
        Describe this image concisely. Keep all sports/betting details: team names, leagues, odds, scores, stakes.
        Describe the layout, colors, and style.
        Ignore any usernames (like @cappersfree) and logos (like CF).
        Output a short description suitable for an image generation model to recreate the image without watermarks.
        """

        max_retries = 4
        base_delay = 2
        for attempt in range(max_retries):
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": vision_prompt},
                                {"type": "image_url", "image_url": {"url": data_url}}
                            ]
                        }
                    ],
                    max_tokens=300
                )
                description = response.choices[0].message.content.strip()
                print(f"✅ Vision 描述: {description[:150]}...")
                return description
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    wait = base_delay * (2 ** attempt)
                    print(f"⚠️ 速率限制 (429)。第 {attempt+1}/{max_retries} 次重试，等待 {wait}s")
                    await asyncio.sleep(wait)
                else:
                    print(f"❌ Vision 错误: {e}")
                    return None
        print("❌ Vision 重试全部失败")
        return None
    except Exception as e:
        print(f"❌ Vision 准备错误: {e}")
        return None

async def generate_image_from_description(prompt: str) -> str:
    """使用 OpenAI DALL-E（如可用），否则回退到 Replicate SDXL。"""
    global _last_generation_call

    if not prompt or len(prompt) < 10:
        return None

    # 强制冷却
    now = time.time()
    elapsed = now - _last_generation_call
    if elapsed < _generation_cooldown:
        wait = _generation_cooldown - elapsed
        print(f"⏳ 等待 {wait:.1f}s 后进行下一次生成...")
        await asyncio.sleep(wait)
    _last_generation_call = time.time()

    # 清理提示词
    clean_prompt = re.sub(r'[^\x00-\x7F]+', '', prompt)
    if len(clean_prompt) > 300:
        clean_prompt = clean_prompt[:300] + "..."
    final_prompt = f"Recreate this image without any watermarks or logos: {clean_prompt}"

    # 首先尝试 OpenAI DALL-E
    try:
        print(f"🎨 尝试 OpenAI DALL-E...")
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=final_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        return response.data[0].url
    except Exception as e:
        print(f"❌ DALL-E 失败: {e}. 回退到 Replicate SDXL...")
        # 回退到 Replicate SDXL
        replicate_api_token = os.getenv("REPLICATE_API_TOKEN")
        if not replicate_api_token:
            print("⚠️ 未设置 REPLICATE_API_TOKEN。无法生成图像。")
            return None
        try:
            # 使用 SDXL 模型
            output = replicate.run(
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={
                    "prompt": final_prompt,
                    "width": 1024,
                    "height": 1024,
                    "num_outputs": 1,
                    "scheduler": "K_EULER",
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5
                }
            )
            # output 是一个 URL 列表
            if output and len(output) > 0:
                return output[0]
            else:
                return None
        except Exception as e2:
            print(f"❌ Replicate SDXL 错误: {e2}")
            return None

async def regenerate_image_from_bytes(image_bytes: bytes) -> str:
    description = await describe_image_bytes(image_bytes)
    if not description:
        return None
    return await generate_image_from_description(description)
