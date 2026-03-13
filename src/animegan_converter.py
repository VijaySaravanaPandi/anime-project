import os
import cv2
import numpy as np
import subprocess

# Paths
BASE_PATH   = "C:\\Users\\vijay\\OneDrive\\Desktop\\anime-project"
TEMP_PATH   = BASE_PATH + "\\temp"
FFMPEG_PATH = "C:\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin\\ffmpeg.exe"

print("✅ Enhanced OpenCV Anime Converter Ready!")
print(f"   OpenCV version : {cv2.__version__}")

# ── Ultra Quality Cartoon Effect ────────────────────────────────
def process_frame_anime(frame):

    # Step 1 - Resize to HD
    frame = cv2.resize(frame, (1280, 720))

    # Step 2 - Color enhancement
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l       = clahe.apply(l)
    lab     = cv2.merge((l, a, b))
    frame   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Step 3 - Fast bilateral filter (reduced iterations)
    smooth = frame
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, 7, 50, 50)

    # Step 4 - Edge detection
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges     = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 2
    )
    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Step 5 - Combine
    anime = cv2.bitwise_and(smooth, edges_col)

    # Step 6 - Boost saturation
    hsv     = cv2.cvtColor(anime,
                cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    s       = np.clip(s * 2.0, 0, 255)
    v       = np.clip(v * 1.1, 0, 255)
    hsv     = cv2.merge([h, s, v]).astype("uint8")
    anime   = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Step 7 - Sharpen
    kernel = np.array([[ 0, -1,  0],
                        [-1,  5, -1],
                        [ 0, -1,  0]])
    anime  = cv2.filter2D(anime, -1, kernel)

    return anime

# ── Convert Single Video ─────────────────────────────────────────
def convert_video_to_anime(input_path, output_path):
    print(f"\n   🎨 Converting: {os.path.basename(input_path)}")

    cap    = cv2.VideoCapture(input_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temp   = output_path.replace(".mp4", "_tmp.mp4")

    out = cv2.VideoWriter(
        temp,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (1280, 720)
    )

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        anime_frame = process_frame_anime(frame)
        out.write(anime_frame)
        frame_count += 1

        if frame_count % 30 == 0:
            pct = (frame_count/total*100) if total > 0 else 0
            print(f"   ⏳ {frame_count}/{total} ({pct:.0f}%)")

    cap.release()
    out.release()

    # Add audio back with high quality encoding
    command = [
        FFMPEG_PATH, "-y",
        "-i", temp,
        "-i", input_path,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-crf", "16",
        "-preset", "slow",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path
    ]
    subprocess.run(command, capture_output=True)

    if os.path.exists(temp):
        os.remove(temp)

    size_mb = os.path.getsize(output_path) / (1024*1024) \
              if os.path.exists(output_path) else 0
    print(f"   ✅ Done! Size: {size_mb:.1f} MB")
    return output_path

# ── Convert All Clips ────────────────────────────────────────────
def convert_all_clips_anime(num_clips):
    print(f"\n🔵 Converting {num_clips} clips to anime style...")
    anime_clips = []

    for i in range(num_clips):
        input_path  = os.path.join(TEMP_PATH, f"clip_{i}.mp4")
        output_path = os.path.join(TEMP_PATH, f"anime_{i}.mp4")

        if not os.path.exists(input_path):
            print(f"   ❌ Not found: clip_{i}.mp4")
            continue

        result = convert_video_to_anime(input_path, output_path)
        if result and os.path.exists(result):
            anime_clips.append(result)

    print(f"\n✅ Total anime clips : {len(anime_clips)}")
    return anime_clips

# ── Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🎨 ULTRA QUALITY ANIME CONVERTER TEST")
    print("=" * 60)

    # Count clips
    clip_count = sum(
        1 for i in range(20)
        if os.path.exists(os.path.join(TEMP_PATH, f"clip_{i}.mp4"))
    )
    print(f"   Found {clip_count} clips in temp folder")

    if clip_count > 0:
        anime_clips = convert_all_clips_anime(clip_count)
        print(f"\n✅ Conversion complete!")
        print(f"✅ Clips saved in: {TEMP_PATH}")
    else:
        # Test with single frame
        print("   No clips found — Testing with single frame...")
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        test_frame[:] = (100, 150, 200)
        result = process_frame_anime(test_frame)
        print(f"   ✅ Frame processed: {result.shape}")
        print("   ✅ Anime converter working correctly!")