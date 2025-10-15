import os
import cv2
from tqdm import tqdm

# Path to video folders
base_dir = "data/Real-life_Deception_Detection_2016/Clips"
output_dir = "data/processed/faces"
categories = {"Truthful": "truth", "Deceptive": "lie"}

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create output folders
for label in categories.values():
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# Process each category
for category_folder, label_name in categories.items():
    category_path = os.path.join(base_dir, category_folder)
    output_path = os.path.join(output_dir, label_name)

    print(f"📂 Processing: {category_folder}")
    for filename in tqdm(os.listdir(category_path)):
        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(category_path, filename)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                save_path = os.path.join(output_path, f"{filename[:-4]}_f{frame_count}.jpg")
                cv2.imwrite(save_path, face_img)
                saved_count += 1
                break  # Save only the first face detected per frame

            frame_count += 1

        cap.release()
        print(f"✅ {filename}: saved {saved_count} face frames")

print("🎉 All face extraction complete!")
