# ML_Based-RealTime_Expression_Tracker
🧠 Real-time Facial Expression Recognition using OpenCV, Mediapipe, and Deep Learning — detects human emotions live via webcam.

# 😃 Real-Time Expression Tracker

A **real-time facial expression recognition system** that uses **computer vision** and **deep learning** to detect human emotions live from a webcam feed.  
It identifies expressions like **happy, sad, angry, surprised, neutral, fear, and disgust** — and displays the detected emotion directly on the live video window.

---

## 🚀 Overview

This project leverages **OpenCV** and **Deep Learning** to perform live emotion detection.  
It processes real-time webcam frames, detects faces, extracts key facial features, and classifies the expression using a trained model.  
Useful for:
- Human-computer interaction  
- Mental state analysis  
- AI-based interview or mood-tracking systems  

---

## ✨ Features

✅ Real-time face detection using **OpenCV**  
✅ Emotion classification (Happy, Sad, Angry, etc.)  
✅ Works smoothly on CPU — no GPU required  
✅ Lightweight, fast, and easy to run  
✅ Modular and ready to integrate with other AI systems  

---

## 🧠 Tech Stack

| Category | Technologies Used |
|-----------|------------------|
| **Language** | Python 3.8+ |
| **Computer Vision** | OpenCV, Mediapipe |
| **Machine Learning** | TensorFlow / PyTorch (depending on your model) |
| **Libraries** | NumPy, Pandas, Scikit-learn, Matplotlib |
| **IDE (Recommended)** | PyCharm |

---

## 📁 Project Structure

LieDetectionProject/
│
├── scripts/
│ ├── realtime_lie_detection.py # Main script (expression tracker)
│ ├── face_detector.py # Face detection module (optional)
│ ├── emotion_model.py # Model loading and prediction
│ └── utils.py # Helper functions
│
├── models/
│ └── expression_model.pth # Trained emotion recognition model
│
├── requirements.txt
├── README.md
└── venv/

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository
git clone https://github.com/<your-username>/RealTime-Expression-Tracker.git
cd RealTime-Expression-Tracker

2️⃣ Create and Activate Virtual Environment
python -m venv venv
.\venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

Or install manually:
pip install opencv-python mediapipe torch torchvision numpy pandas matplotlib scikit-learn

▶️ Running the Project
Make sure your webcam is connected, then run:
python scripts/realtime_lie_detection.py

📊 Detected Emotions
The model can identify the following emotions:

😀 Happy

😢 Sad

😠 Angry

😱 Surprised

😐 Neutral

😨 Fear

🤢 Disgust

Each detected emotion is displayed in real time on the video frame.

🧩 Requirements File Example
If not created yet, here’s a sample requirements.txt:
opencv-python
mediapipe
torch
torchvision
numpy
pandas
scikit-learn
matplotlib

💡 Future Improvements
🚀 Add support for multiple faces
🧠 Improve accuracy with CNN / transformer-based models
🎙️ Combine facial + voice emotion detection
🌐 Deploy with Flask/Streamlit dashboard

👨‍💻 Author
Developed by: Shiv Chandra Bind
📧 Email: bdrdshivchandra9125@gmail.com
💻 Passionate about Computer Vision & Real-Time AI Systems

🪪 License
This project is released under the MIT License — free for personal and educational use.
