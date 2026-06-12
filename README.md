🎭 Emotion-Aware Object Detection

Real-time object detection + facial emotion recognition using YOLOv8 and DeepFace

Show Image

Show Image

Show Image

Show Image

Show Image

📌 About
This project is built for ELC Activity 2025-26 under the topic Real-Time Applications Based on Computer Vision.
It combines two powerful AI systems into one unified real-time pipeline:

YOLOv8n — detects 80 types of objects (person, laptop, chair, etc.)
DeepFace — recognises 7 facial emotions (happy, sad, angry, surprise, fear, disgust, neutral)

The result is a system that doesn't just see what is in a scene — it understands how people are feeling.

🎬 Demo
📷 Webcam Input
      ↓
⚡ YOLOv8n  →  Detects objects with bounding boxes
      ↓
👁️  Haar Cascade  →  Finds faces in the frame
      ↓
🧠 DeepFace  →  Classifies emotion per face
      ↓
🖥️  Annotated Output with live HUD

✨ Features

🔴 Real-time detection at 18–25 FPS on standard CPU
🎯 80 object classes from the COCO dataset
😊 7 emotion categories with probability bar charts
🖥️ Live HUD showing FPS, frame count, object count, face count
📸 Screenshot with S key, quit with Q key
💾 Save output as AVI video file
⚡ Frame-skip optimisation — emotion runs every 3rd frame for performance
🔁 Graceful fallback — works even without DeepFace or YOLO installed


🗂️ Project Structure
emotion_detection/
│
├── src/
│   ├── emotion_object_detector.py   # Main pipeline — run this
│   └── evaluate.py                  # Metrics: accuracy, F1, confusion matrix
│
├── results/                         # Screenshots and output videos saved here
├── requirements.txt                 # All dependencies
└── README.md                        # This file

🚀 Getting Started
1. Clone the repository
bashgit clone https://github.com/yourusername/emotion-aware-detection.git
cd emotion-aware-detection
2. Install dependencies
bashpip install -r requirements.txt

First run will auto-download the YOLOv8n model weights (~6MB)

3. Run on webcam
bashpython src/emotion_object_detector.py
4. Run on a video file
bashpython src/emotion_object_detector.py --source path/to/video.mp4
5. Save the output video
bashpython src/emotion_object_detector.py --source 0 --save --output results/demo.avi

⌨️ Keyboard Controls
KeyActionQQuit the applicationSSave screenshot to results/ folder

📊 Evaluate the Emotion Classifier
Prepare a folder like this:
eval_data/
  happy/    → *.jpg images
  sad/      → *.jpg images
  angry/    → *.jpg images
  ...
Then run:
bashpython src/evaluate.py --data_dir eval_data/
Output includes per-class precision, recall, F1-score, accuracy, and confusion matrix.

📈 Results
MetricValueEmotion Accuracy (macro)~67%macro Precision0.67macro Recall0.62macro F1-Score0.64Object Detection mAP5037.3 (COCO)Average FPS (CPU)18 – 25Average FPS (GPU)45 – 60
Per-class Emotion Accuracy
EmotionAccuracy😊 Happy78%😐 Neutral72%😲 Surprise70%😢 Sad64%😠 Angry58%🤢 Disgust55%😨 Fear52%

🛠️ Tech Stack
ToolVersionPurposePython3.10+Core languageOpenCV4.8+Capture, face detection, renderingUltralytics YOLOv88.0+Object detectionDeepFace0.0.79+Emotion recognitionPyTorch2.0+DL backendNumPy1.24+Numerical opsscikit-learn1.3+Evaluation metrics

⚙️ How It Works

Each frame is captured from webcam or video
YOLOv8n runs on every frame and draws object boxes
Haar Cascade detects face locations every frame
DeepFace analyses each face crop every 3rd frame
Emotion result is cached for skipped frames
All annotations are rendered and displayed live

The frame-skipping trick cuts CPU load by ~60% while keeping the video smooth.

🔮 Future Improvements

 Replace Haar Cascade with MTCNN for better face detection
 Temporal smoothing with Kalman filter to reduce emotion flickering
 Multi-person tracking with ByteTrack / DeepSORT
 Edge deployment on Raspberry Pi 5 / Jetson Nano (INT8 quantised)
 Audio-visual fusion — combine speech tone + face expression


📚 References

Jocher et al. (2023). Ultralytics YOLOv8
Serengil, S.I. (2021). DeepFace
Goodfellow et al. (2013). Challenges in Representation Learning: FER-2013. ICML.
Bradski, G. (2000). The OpenCV Library.
