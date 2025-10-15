import os
import subprocess

# --- Set your main paths ---
video_root = os.path.join('data', 'Real-life_Deception_Detection_2016', 'Clips')
output_root = os.path.join('data', 'audio')

# --- Folder categories ---
categories = ['Truthful', 'Deceptive']

# --- Loop through each folder ---
for category in categories:
    input_dir = os.path.join(video_root, category)
    output_dir = os.path.join(output_root, category, 'raw')
    os.makedirs(output_dir, exist_ok=True)

    for video_file in os.listdir(input_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(input_dir, video_file)
            audio_filename = os.path.splitext(video_file)[0] + '.wav'
            audio_output = os.path.join(output_dir, audio_filename)

            # FFmpeg command
            command = [
                'ffmpeg', '-y',  # -y to auto-overwrite
                '-i', video_path,
                '-vn',  # no video
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # sample rate
                '-ac', '1',  # mono
                audio_output
            ]

            print(f"[+] Processing: {video_path}")
            try:
                subprocess.run(command, check=True)
                print(f"[✓] Saved audio to: {audio_output}\n")
            except subprocess.CalledProcessError as e:
                print(f"[!] FFmpeg failed on: {video_path}\n{e}")
