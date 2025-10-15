import os
import cv2
import mediapipe as mp

# Paths
RAW_LPW_PATH = "data/lpw"
PROCESSED_LPW_PATH = "data/processed/lpw"
os.makedirs(PROCESSED_LPW_PATH, exist_ok=True)

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.25  # lower if needed
)

# Loop through subjects
for subject in os.listdir(RAW_LPW_PATH):
    subject_path = os.path.join(RAW_LPW_PATH, subject)
    if not os.path.isdir(subject_path):
        continue

    video_files = [f for f in os.listdir(subject_path) if f.endswith(('.avi', '.mp4', '.mov'))]
    if not video_files:
        print(f"❌ No video found for subject {subject}")
        continue

    saved = 0
    subject_output = os.path.join(PROCESSED_LPW_PATH, subject)
    os.makedirs(subject_output, exist_ok=True)

    for video_file in video_files:
        video_path = os.path.join(subject_path, video_file)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 10:
            print(f"⚠️ Skipping video {video_file} (subject {subject}): too few frames ({total_frames})")
            cap.release()
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                out_path = os.path.join(subject_output, f"{subject}_{saved:05d}.jpg")
                cv2.imwrite(out_path, frame)
                saved += 1

        cap.release()

    print(f"✅ Subject {subject}: saved {saved} frames")

print("🎉 LPW preprocessing complete!")
