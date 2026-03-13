import requests
import os
import time
import json

# Your HuggingFace token (Set this in your environment variables for security)
HF_TOKEN  = os.environ.get("HF_TOKEN", "your_huggingface_token_here")
TEMP_PATH = "C:\\Users\\vijay\\OneDrive\\Desktop\\anime-project\\temp"

headers = {
    "Authorization" : f"Bearer {HF_TOKEN}",
    "Content-Type"  : "application/json"
}

# ── Check Available Endpoints ────────────────────────────────────
def check_endpoints():
    print("🔵 Checking available endpoints...")

    urls = [
        "https://router.huggingface.co/hf-inference/models/THUDM/CogVideoX-5b",
        "https://router.huggingface.co/hf-inference/models/THUDM/CogVideoX-2b",
        "https://router.huggingface.co/hf-inference/models/ali-vilab/text-to-video-ms-1.7b",
        "https://router.huggingface.co/hf-inference/models/damo-vilab/text-to-video-ms-1.7b",
    ]

    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            print(f"   {r.status_code} → {url}")
        except Exception as e:
            print(f"   ERROR → {url}: {e}")

# ── Generate Video ───────────────────────────────────────────────
def generate_video(prompt, output_path, api_url, retries=3):
    print(f"\n   🎬 Prompt : {prompt[:60]}...")
    print(f"   🔗 URL    : {api_url}")

    styled_prompt = (
        f"Pixar Disney 3D animation style, {prompt}, "
        f"colorful vibrant, smooth animation, "
        f"high quality render, cinematic"
    )

    payload = {"inputs": styled_prompt}

    for attempt in range(retries):
        print(f"\n   ⏳ Attempt {attempt+1}/{retries}...")
        print(f"   ⏳ Sending request (may take 2-5 mins)...")

        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=600
            )

            print(f"   Status : {response.status_code}")
            print(f"   Headers: {dict(response.headers).get('content-type','')}")

            if response.status_code == 200:
                content_type = response.headers.get('content-type','')
                if 'video' in content_type or 'octet' in content_type:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    size_mb = os.path.getsize(output_path)/(1024*1024)
                    print(f"   ✅ Video saved! Size: {size_mb:.1f} MB")
                    return output_path
                else:
                    print(f"   Response: {response.text[:300]}")

            elif response.status_code == 503:
                wait = 30 * (attempt + 1)
                print(f"   ⏳ Model loading... waiting {wait}s")
                time.sleep(wait)

            elif response.status_code == 401:
                print(f"   ❌ Invalid token!")
                print(f"   {response.text[:200]}")
                return None

            elif response.status_code == 402:
                print(f"   ❌ Payment required!")
                print(f"   {response.text[:200]}")
                return None

            elif response.status_code == 429:
                print(f"   ⏳ Rate limited... waiting 60s")
                time.sleep(60)

            else:
                print(f"   ❌ Error {response.status_code}")
                print(f"   {response.text[:300]}")
                time.sleep(10)

        except requests.Timeout:
            print(f"   ⏳ Timeout... retrying")
        except Exception as e:
            print(f"   ❌ Exception: {e}")

    return None

# ── Try All Models ───────────────────────────────────────────────
def try_all_models(prompt, output_path):
    models = [
        {
            "name" : "CogVideoX-5b",
            "url"  : "https://router.huggingface.co/hf-inference/models/THUDM/CogVideoX-5b"
        },
        {
            "name" : "CogVideoX-2b",
            "url"  : "https://router.huggingface.co/hf-inference/models/THUDM/CogVideoX-2b"
        },
        {
            "name" : "text-to-video-ms",
            "url"  : "https://router.huggingface.co/hf-inference/models/ali-vilab/text-to-video-ms-1.7b"
        },
        {
            "name" : "damo-text-to-video",
            "url"  : "https://router.huggingface.co/hf-inference/models/damo-vilab/text-to-video-ms-1.7b"
        }
    ]

    for m in models:
        print(f"\n{'='*50}")
        print(f"🔵 Trying model: {m['name']}")
        print(f"{'='*50}")
        result = generate_video(prompt, output_path, m['url'])
        if result:
            print(f"\n🎉 SUCCESS with {m['name']}!")
            return result, m['name']
        else:
            print(f"   ⚠️ {m['name']} failed, trying next...")

    return None, None

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🎬 COGVIDEOX API TEST")
    print("=" * 60)

    os.makedirs(TEMP_PATH, exist_ok=True)

    # First check endpoints
    check_endpoints()

    test_prompt = "A hero picks up his sword in a magical forest"
    output_path = os.path.join(TEMP_PATH, "test_cogvideo.mp4")

    print(f"\n{'='*60}")
    print(f"🔵 Starting video generation...")
    print(f"   Story  : {test_prompt}")
    print(f"   Style  : Pixar Disney 3D animation")
    print(f"   Output : {output_path}")
    print(f"{'='*60}")

    result, model_name = try_all_models(test_prompt, output_path)

    if result:
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS!")
        print(f"✅ Model used  : {model_name}")
        print(f"✅ Video saved : {result}")
        print(f"\n🎉 Open this file to see Pixar style video:")
        print(f"   {output_path}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"❌ All models failed!")
        print(f"   Possible reasons:")
        print(f"   1. Models need PRO subscription")
        print(f"   2. API endpoints changed")
        print(f"   3. Rate limit exceeded")
        print(f"{'='*60}")