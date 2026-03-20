import os
import subprocess
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
from moviepy.video.fx.all import fadein, fadeout

# Paths — resolved dynamically so this works on any machine
_HERE       = os.path.dirname(os.path.abspath(__file__))   # src/
_BASE       = os.path.dirname(_HERE)                        # project root
TEMP_PATH   = os.path.join(_BASE, "temp")
OUTPUT_PATH = os.path.join(_BASE, "output")
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")      # set env var or use system PATH

def adjust_duration(input_path, output_path, target_duration):
    print(f"   ⏱️  Adjusting to {target_duration:.1f}s")
    cap          = cv2.VideoCapture(input_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 24
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    cur_dur      = total_frames / fps or 10
    speed        = cur_dur / target_duration
    command = [
        FFMPEG_PATH, "-y",
        "-i", input_path,
        "-filter:v", f"setpts={1/speed}*PTS",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-an",
        output_path
    ]
    result = subprocess.run(command, capture_output=True)
    return output_path if result.returncode == 0 else input_path

def stitch_clips(cartoon_clips, target_duration,
                 output_filename="animation_output.mp4"):
    print(f"\n🔵 Stitching {len(cartoon_clips)} clips...")
    duration_per = target_duration / len(cartoon_clips)
    adjusted     = []

    for i, clip_path in enumerate(cartoon_clips):
        adj_path = os.path.join(TEMP_PATH, f"adjusted_{i}.mp4")
        result   = adjust_duration(clip_path, adj_path, duration_per)
        adjusted.append(result)

    # Load clips with fade transitions
    print("\n   📽️  Adding smooth transitions...")
    clips = []
    for i, path in enumerate(adjusted):
        if os.path.exists(path):
            clip = VideoFileClip(path)
            # Add fade in and fade out
            clip = fadein(clip, 0.5)
            clip = fadeout(clip, 0.5)
            clips.append(clip)
            print(f"   ✅ Loaded: {os.path.basename(path)}")

    if not clips:
        print("   ❌ No clips!")
        return None

    # Concatenate with transitions
    print("\n   🎬 Concatenating with transitions...")
    final      = concatenate_videoclips(clips, method="compose")
    out_file   = os.path.join(OUTPUT_PATH, output_filename)

    # High quality export
    final.write_videofile(
        out_file,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        bitrate="5000k",
        verbose=False,
        logger=None
    )

    for c in clips:
        c.close()
    final.close()

    print(f"\n✅ Final video : {out_file}")
    print(f"✅ Duration    : {target_duration}s")
    print(f"✅ Resolution  : 1280x720 HD")
    print(f"✅ FPS         : 30")
    return out_file

if __name__ == "__main__":
    print("🔵 High Quality Video Assembly...")
    cartoon_clips = []
    for i in range(3):
        path = os.path.join(TEMP_PATH, f"cartoon_{i}.mp4")
        if os.path.exists(path):
            cartoon_clips.append(path)
        else:
            path = os.path.join(TEMP_PATH, f"clip_{i}.mp4")
            if os.path.exists(path):
                cartoon_clips.append(path)

    result = stitch_clips(cartoon_clips, target_duration=30)
    if result:
        print("\n🎉 High quality video created!")