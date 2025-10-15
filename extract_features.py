import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.preprocessing import StandardScaler

# Define directories for the Truthful and Deceptive audio files
input_audio_dirs = [
    "D:\\Projects\\LieDetectionProject\\data\\audio\\Truthful\\raw",
    "D:\\Projects\\LieDetectionProject\\data\\audio\\Deceptive\\raw"
]

# Output directory where you want to save the features
output_features_dir = "D:\\Projects\\LieDetectionProject\\data\\audio\\features"

# Create output directory if it doesn't exist
if not os.path.exists(output_features_dir):
    os.makedirs(output_features_dir)

# Define the function to extract features from the audio files
def extract_audio_features(audio_path):
    try:
        # Load the audio file with librosa
        y, sr = librosa.load(audio_path, sr=16000)

        # Extract features such as MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        # Extract features such as chroma, spectral contrast, etc.
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_scaled = np.mean(chroma.T, axis=0)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_scaled = np.mean(spectral_contrast.T, axis=0)

        # Combine the extracted features into a single feature vector
        features = np.hstack([mfccs_scaled, chroma_scaled, spectral_contrast_scaled])

        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Process each folder (Truthful and Deceptive)
for input_audio_dir in input_audio_dirs:
    for filename in os.listdir(input_audio_dir):
        if filename.endswith('.wav'):
            audio_path = os.path.join(input_audio_dir, filename)

            # Extract features from the audio file
            features = extract_audio_features(audio_path)

            if features is not None:
                # Create a DataFrame to hold the features
                features_df = pd.DataFrame([features])

                # Add a label to distinguish between Truthful and Deceptive
                label = 'Truthful' if 'truth' in input_audio_dir.lower() else 'Deceptive'
                features_df['label'] = label

                # Save the features to a CSV file
                output_filename = f"{os.path.splitext(filename)[0]}_features.csv"
                output_file = os.path.join(output_features_dir, output_filename)
                features_df.to_csv(output_file, index=False)

                print(f"Features extracted and saved for {filename}")
            else:
                print(f"Skipping {filename} due to extraction error.")
quit()