import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision import models
import os

# ✅ Load trained model
base_dir = os.path.dirname(os.path.dirname(__file__))  # Goes one level up from 'scripts'
model_path = os.path.join(base_dir, 'saved_models', 'lie_detection_model.pth')
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # 4 classes: casme2, samm, lpw, ubfc
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ✅ Label map (you can change these to "Lie" / "Truth" etc. based on how your classes were mapped)
class_names = ['neutral', 'happy', 'sad', 'surprised']  # Update this if you have actual emotion labels

# ✅ Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ✅ Open webcam
cap = cv2.VideoCapture(0)
print("📸 Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract and preprocess face ROI
        face = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        input_tensor = transform(face_pil).unsqueeze(0)  # Add batch dim

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            pred_label = class_names[predicted.item()]
            confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()

        # Draw prediction on frame
        label_text = f"{pred_label} ({confidence*100:.1f}%)"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("🎥 Lie Detection Live Feed", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Clean up
cap.release()
cv2.destroyAllWindows()
