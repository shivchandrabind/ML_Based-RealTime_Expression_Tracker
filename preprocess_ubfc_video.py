import cv2
import os
from pathlib import Path
from tqdm import tqdm

# Path to your .avi video
VIDEO_PATH = Path("data/UBFC_DATASET/DATASET_1/10-gt/vid.avi")  # replace if filename is different
OUTPUT_DIR = Path("data/processed/ubfc/subject_10")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open video
cap = cv2.VideoCapture(str(VIDEO_PATH))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_idx = 0
saved_frames = 0

print(f"📽️ Processing video: {VIDEO_PATH.name} | Total frames: {frame_count}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_path = OUTPUT_DIR / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(face_path), face)
        saved_frames += 1
        break  # Save only one face per frame

    frame_idx += 1

cap.release()
print(f"✅ Done! Total saved face frames: {saved_frames}")
