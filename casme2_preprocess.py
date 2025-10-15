import os
import cv2
import mediapipe as mp

# Correct path to your CASME 2 data folder
CASME2_DIR = "data/CASME2 Preprocessed v2"
OUTPUT_DIR = "data/processed/casme2"

# MediaPipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

def process_emotion_folder(emotion_folder):
    emotion_path = os.path.join(CASME2_DIR, emotion_folder)
    output_emotion_path = os.path.join(OUTPUT_DIR, emotion_folder)
    os.makedirs(output_emotion_path, exist_ok=True)

    for filename in os.listdir(emotion_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue

        img_path = os.path.join(emotion_path, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                for lm in face_landmarks.landmark:
                    h, w, _ = image.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

            out_path = os.path.join(output_emotion_path, filename)
            cv2.imwrite(out_path, image)

def run():
    emotions = os.listdir(CASME2_DIR)
    for emotion in emotions:
        print(f"Processing emotion category: {emotion}")
        process_emotion_folder(emotion)

if __name__ == "__main__":
    run()
