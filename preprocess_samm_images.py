import cv2
import os
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm

# Paths
ROOT = Path(__file__).resolve().parent.parent
SAMM_RAW = ROOT / "data" / "SAMM_v1"
SAMM_PROCESSED = ROOT / "data" / "processed" / "samm"
SAMM_PROCESSED.mkdir(parents=True, exist_ok=True)

# Face detector
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.3)

# Loop through emotions
for emotion_folder in SAMM_RAW.iterdir():
    if not emotion_folder.is_dir():
        continue

    emotion = emotion_folder.name
    output_emotion_path = SAMM_PROCESSED / emotion
    output_emotion_path.mkdir(parents=True, exist_ok=True)

    # Loop through images
    for image_file in tqdm(list(emotion_folder.glob("*.jpg")), desc=f"Processing {emotion}"):
        image = cv2.imread(str(image_file))
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                face = image[y:y+bh, x:x+bw]
                if face.size == 0:
                    continue

                out_path = output_emotion_path / image_file.name
                cv2.imwrite(str(out_path), face)
                break

print("✅ SAMM_v1 face extraction complete!")
