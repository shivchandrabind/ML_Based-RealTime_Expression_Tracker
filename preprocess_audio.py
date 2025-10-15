import os
import librosa
import soundfile as sf
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# Input and output paths
video_dir = "data/Real-life_Deception_Detection_2016/Clips"
output_dir = "data/processed/audio_features"
categories = {"Truthful": "truth", "Deceptive": "lie"}

# Create output dirs
for label in categories.values():
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

def extract_mfcc_from_video(video_path, save_path):
    try:
        # Extract audio from video
        video = VideoFileClip(video_path)
        audio_path = save_path.replace(".npy", ".wav")
        video.audio.write_audiofile(audio_path, logger=None)

        # Load audio
        y, sr = librosa.load(audio_path, sr=None)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Save MFCC
        np.save(save_path, mfcc_mean)

        # Clean up
        os.remove(audio_path)
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Process files
for folder, label_name in categories.items():
    folder_path = os.path.join(video_dir, folder)
    output_path = os.path.join(output_dir, label_name)

    print(f"🔊 Processing {folder} videos...")
    for file in tqdm(os.listdir(folder_path)):
        if not file.endswith(".mp4"):
            continue

        video_path = os.path.join(folder_path, file)
        save_path = os.path.join(output_path, file.replace(".mp4", ".npy"))
        extract_mfcc_from_video(video_path, save_path)

print("✅ Audio feature extraction complete.")
