import os
from openai import OpenAI

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

async def rewrite_text(original_text: str) -> str:
    if not original_text or len(original_text.strip()) < 5:
        return original_text

    # Replace @usenane with @PrimeAnalysiss (case‑insensitive)
    # Use regex to replace whole words
    import re
    modified_text = re.sub(r'@usenane\b', '@PrimeAnalysiss', original_text, flags=re.IGNORECASE)

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a content rewriter. Rephrase the provided text to make it unique while keeping all key information, tone, and meaning. Preserve all @mentions exactly as written."},
                {"role": "user", "content": modified_text}
            ],
            temperature=0.8,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"DeepSeek error: {e}")
        return modified_text  # fallback to modified original
