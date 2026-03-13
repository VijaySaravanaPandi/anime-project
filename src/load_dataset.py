import json
import pandas as pd
import os

# Correct paths
DATASET_PATH    = "C:\\Users\\vijay\\OneDrive\\Desktop\\anime-project\\datasets-project\\data\\MSRVTT\\MSRVTT"
VIDEOS_PATH     = os.path.join(DATASET_PATH, "videos", "all")
ANNOTATION_PATH = os.path.join(DATASET_PATH, "annotation")
annotation_file = os.path.join(ANNOTATION_PATH, "MSR_VTT.json")

print("Loading dataset...")

with open(annotation_file, "r") as f:
    data = json.load(f)

# Convert to dataframes using correct keys
videos_df   = pd.DataFrame(data['images'])       # images = videos
captions_df = pd.DataFrame(data['annotations'])  # annotations = captions

# Rename columns to match
videos_df.rename(columns={'id': 'video_id'}, inplace=True)
captions_df.rename(columns={'image_id': 'video_id'}, inplace=True)

# Merge videos with captions
merged_df = pd.merge(captions_df, videos_df, on='video_id')

# Save merged data
merged_df.to_csv(
    "C:\\Users\\vijay\\OneDrive\\Desktop\\anime-project\\src\\merged_data.csv",
    index=False
)

# Display info
print(f"✅ Total Videos     : {len(videos_df)}")
print(f"✅ Total Captions   : {len(captions_df)}")
print(f"✅ Merged Rows      : {len(merged_df)}")
print(f"\n📌 Sample Data:")
print(merged_df[['video_id', 'caption']].head(5))
print(f"\n✅ Videos Path      : {VIDEOS_PATH}")
print("✅ Dataset loaded and saved successfully!")