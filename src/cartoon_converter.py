import cv2
import os
import subprocess
import numpy as np

# Paths — resolved dynamically so this works on any machine
_HERE       = os.path.dirname(os.path.abspath(__file__))   # src/
_BASE       = os.path.dirname(_HERE)                        # project root
TEMP_PATH   = os.path.join(_BASE, "temp")
OUTPUT_PATH = os.path.join(_BASE, "output")
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")      # set env var or use system PATH

# HIGH QUALITY cartoon effect
def apply_cartoon_effect(frame):
    # Step 1 - Resize to higher resolution
    frame = cv2.resize(frame, (1280, 720))

    # Step 2 - Color enhancement
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l     = clahe.apply(l)
    lab   = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Step 3 - Strong bilateral filter for smooth cartoon look
    color = frame
    for _ in range(4):
        color = cv2.bilateralFilter(color, 9, 75, 75)

    # Step 4 - Edge detection
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges     = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 2
    )

    # Step 5 - Convert edges to 3 channel
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Step 6 - Combine color with edges
    cartoon = cv2.bitwise_and(color, edges_colored)

    # Step 7 - Boost saturation for vibrant cartoon look
    hsv        = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v    = cv2.split(hsv)
    s          = np.clip(s * 1.8, 0, 255)
    v          = np.clip(v * 1.1, 0, 255)
    hsv        = cv2.merge([h, s, v]).astype("uint8")
    cartoon    = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return cartoon

# Convert single video to high quality cartoon
def convert_to_cartoon(input_path, output_path):
    print(f"   🎨 Converting: {os.path.basename(input_path)}")

    cap    = cv2.VideoCapture(input_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    temp   = output_path.replace(".mp4", "_temp.mp4")

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
        cartoon_frame = apply_cartoon_effect(frame)
        out.write(cartoon_frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   ⏳ Frames: {frame_count}/{total}")

    cap.release()
    out.release()

    # Add audio + high quality encoding
    command = [
        FFMPEG_PATH, "-y",
        "-i", temp,
        "-i", input_path,
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path
    ]
    subprocess.run(command, capture_output=True)

    if os.path.exists(temp):
        os.remove(temp)

    print(f"   ✅ Done: {os.path.basename(output_path)}")
    return output_path

def convert_all_clips(num_clips):
    cartoon_clips = []
    for i in range(num_clips):
        input_path  = os.path.join(TEMP_PATH, f"clip_{i}.mp4")
        output_path = os.path.join(TEMP_PATH, f"cartoon_{i}.mp4")
        if not os.path.exists(input_path):
            print(f"   ❌ Not found: clip_{i}.mp4")
            continue
        result = convert_to_cartoon(input_path, output_path)
        if result:
            cartoon_clips.append(output_path)
    return cartoon_clips

if __name__ == "__main__":
    print("🔵 High Quality Cartoon Conversion...")
    cartoon_clips = convert_all_clips(3)
    print(f"\n✅ Total clips  : {len(cartoon_clips)}")
    print("✅ Done!")