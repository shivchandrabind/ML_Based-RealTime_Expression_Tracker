import torch
import torch.nn as nn
import librosa
import numpy as np
import sys

# === Model Definition (same as training) ===
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

# === Load audio and extract MFCC features ===
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return torch.tensor(mfcc_mean, dtype=torch.float32)

# === Load trained model ===
model_path = "saved_models/audio_lie_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dummy input to get input size
dummy = extract_features("data/audio/Deceptive/raw/trial_lie_001.wav")  # or any existing file
model = MLP(input_size=len(dummy))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Predict ===
if len(sys.argv) != 2:
    print("❌ Usage: python predict_audio.py path_to_audio.wav")
    sys.exit(1)

audio_path = sys.argv[1]
features = extract_features(audio_path).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(features)
    probs = torch.nn.functional.softmax(output, dim=1)
    pred = torch.argmax(probs).item()
    label = "Lie" if pred == 1 else "Truth"
    confidence = probs[0][pred].item() * 100

print(f"🎙️ Prediction: {label} ({confidence:.2f}%)")
