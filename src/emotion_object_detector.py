"""
Emotion-Aware Object Detection System
ELC Activity 2025-26 | Computer Vision
--------------------------------------
Detects objects (YOLO) + emotions (DeepFace) in real-time
from webcam or video file, overlays rich annotations.
"""

import cv2
import numpy as np
import time
import argparse
import os
from collections import deque

# ── Optional imports (graceful fallback if not installed) ──────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[WARN] DeepFace not installed. Emotion detection will be simulated.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] Ultralytics not installed. Object detection will use HOG fallback.")


# ── Colour palette ─────────────────────────────────────────────────────────────
EMOTION_COLORS = {
    "happy":    (0,   210, 120),
    "sad":      (200,  80,  30),
    "angry":    (0,    0,  220),
    "surprise": (0,   200, 240),
    "fear":     (130,  0,  180),
    "disgust":  (0,   130,  60),
    "neutral":  (180, 180, 180),
}

EMOTION_ICONS = {
    "happy":    ":-)",
    "sad":      ":-(",
    "angry":    ">:-(",
    "surprise": ":-O",
    "fear":     "D:-",
    "disgust":  ":-P",
    "neutral":  ":-|",
}

OBJ_COLOR  = (255, 200,  50)   # golden for objects
TEXT_COLOR = (255, 255, 255)


# ── Helper utilities ───────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=12, fill=True):
    x1, y1 = pt1
    x2, y2 = pt2
    if fill:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, -1)
        cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(img, (cx, cy), radius, color, thickness)


def draw_label_box(img, text, x, y, color, font_scale=0.55, thickness=1):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 5
    cv2.rectangle(img, (x-pad, y-th-pad*2), (x+tw+pad, y+baseline), color, -1)
    cv2.putText(img, text, (x, y-pad), font, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def emotion_bar(img, emotions: dict, x, y, width=160):
    """Draw a small horizontal bar chart for emotion probabilities."""
    bar_h = 10
    gap   = 3
    font  = cv2.FONT_HERSHEY_SIMPLEX
    for i, (emo, prob) in enumerate(sorted(emotions.items(), key=lambda e: -e[1])[:4]):
        bar_y = y + i * (bar_h + gap)
        filled = int(width * prob / 100)
        col = EMOTION_COLORS.get(emo, (180, 180, 180))
        cv2.rectangle(img, (x, bar_y), (x+width, bar_y+bar_h), (50,50,50), -1)
        cv2.rectangle(img, (x, bar_y), (x+filled, bar_y+bar_h), col, -1)
        cv2.putText(img, f"{emo[:7]} {prob:.0f}%", (x+width+5, bar_y+bar_h-1),
                    font, 0.32, col, 1, cv2.LINE_AA)


def draw_hud(img, fps, frame_count, detections, emotions_found):
    """Top-left heads-up display."""
    h, w = img.shape[:2]
    hud_h = 90
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (300, hud_h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    font  = cv2.FONT_HERSHEY_DUPLEX
    font2 = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "EMOTION-AWARE DETECTOR", (10, 22), font, 0.52,
                (100, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps:5.1f}   Frame: {frame_count:05d}", (10, 46),
                font2, 0.42, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(img, f"Objects: {detections}   Faces: {emotions_found}", (10, 66),
                font2, 0.42, (255, 220, 100), 1, cv2.LINE_AA)
    cv2.putText(img, "Press Q to quit | S to screenshot", (10, 84),
                font2, 0.32, (150, 150, 150), 1, cv2.LINE_AA)


# ── Core detector class ────────────────────────────────────────────────────────
class EmotionAwareDetector:
    def __init__(self, source=0, save_output=False, output_path="results/output.avi"):
        self.source       = source
        self.save_output  = save_output
        self.output_path  = output_path
        self.frame_count  = 0
        self.fps_buffer   = deque(maxlen=30)
        self.screenshot_n = 0
        self.emotion_cache = {}   # face_id -> emotion dict (smoothing)

        # Load YOLO (or HOG fallback)
        if YOLO_AVAILABLE:
            print("[INFO] Loading YOLOv8n model …")
            self.yolo = YOLO("yolov8n.pt")   # auto-downloads on first run
        else:
            print("[INFO] Using HOG people detector as fallback …")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Face detector
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        print("[INFO] Initialisation complete.\n")

    # ── Object detection ───────────────────────────────────────────────────────
    def detect_objects(self, frame):
        """Returns list of (x1,y1,x2,y2,label,conf)."""
        results = []
        if YOLO_AVAILABLE:
            yolo_res = self.yolo(frame, verbose=False, conf=0.35)[0]
            for box in yolo_res.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                label = yolo_res.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                results.append((x1, y1, x2, y2, label, conf))
        else:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes, _ = self.hog.detectMultiScale(gray, winStride=(8,8), padding=(4,4), scale=1.05)
            for (x,y,w,h) in boxes:
                results.append((x, y, x+w, y+h, "person", 0.8))
        return results

    # ── Emotion detection ──────────────────────────────────────────────────────
    def detect_emotions(self, frame, face_rois):
        """Returns dict mapping face_index -> emotion info."""
        out = {}
        for i, (fx, fy, fw, fh) in enumerate(face_rois):
            face_img = frame[fy:fy+fh, fx:fx+fw]
            if face_img.size == 0:
                continue
            if DEEPFACE_AVAILABLE:
                try:
                    res = DeepFace.analyze(face_img, actions=["emotion"],
                                           enforce_detection=False, silent=True)
                    emo_data = res[0] if isinstance(res, list) else res
                    dom  = emo_data["dominant_emotion"]
                    probs = emo_data["emotion"]
                    out[i] = {"dominant": dom, "probs": probs}
                except Exception:
                    out[i] = self._random_emotion()
            else:
                out[i] = self._random_emotion()
        return out

    def _random_emotion(self):
        """Simulated emotion for demo without DeepFace."""
        import random
        emos = list(EMOTION_COLORS.keys())
        dom  = random.choice(emos)
        probs = {e: round(random.uniform(2, 30), 1) for e in emos}
        probs[dom] = round(random.uniform(55, 90), 1)
        total = sum(probs.values())
        probs = {k: v/total*100 for k,v in probs.items()}
        return {"dominant": dom, "probs": probs}

    # ── Frame rendering ────────────────────────────────────────────────────────
    def render(self, frame, obj_boxes, face_rois, emotion_map):
        annotated = frame.copy()

        # Draw object boxes
        for (x1, y1, x2, y2, label, conf) in obj_boxes:
            draw_rounded_rect(annotated, (x1,y1), (x2,y2), OBJ_COLOR, thickness=2, fill=False)
            draw_label_box(annotated, f"{label} {conf:.0%}", x1, y1-4, OBJ_COLOR)

        # Draw face + emotion overlays
        for i, (fx, fy, fw, fh) in enumerate(face_rois):
            emo_info = emotion_map.get(i, {"dominant":"neutral","probs":{}})
            dom      = emo_info["dominant"]
            probs    = emo_info["probs"]
            color    = EMOTION_COLORS.get(dom, (180,180,180))
            icon     = EMOTION_ICONS.get(dom, ":-|")

            # Face bounding box
            draw_rounded_rect(annotated, (fx,fy), (fx+fw,fy+fh),
                              color, thickness=2, fill=False)

            # Emotion label
            tag = f"{icon} {dom.upper()}"
            draw_label_box(annotated, tag, fx, fy-6, color, font_scale=0.6)

            # Probability bars (right of face if space allows)
            bar_x = fx + fw + 8
            bar_y = fy + 4
            if bar_x + 220 < annotated.shape[1] and probs:
                emotion_bar(annotated, probs, bar_x, bar_y)

        return annotated

    # ── Main loop ──────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {self.source}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Stream opened: {w}x{h}")

        writer = None
        if self.save_output:
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(self.output_path, fourcc, 20, (w, h))

        emotion_skip = 3    # run emotion every N frames (performance)
        obj_skip     = 1
        last_obj     = []
        last_faces   = []
        last_emos    = {}

        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1

            # Object detection
            if self.frame_count % obj_skip == 0:
                last_obj = self.detect_objects(frame)

            # Face detection (fast, every frame)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))
            last_faces = list(faces) if len(faces) else last_faces

            # Emotion detection (expensive, every N frames)
            if self.frame_count % emotion_skip == 0:
                last_emos = self.detect_emotions(frame, last_faces)

            # Render
            output = self.render(frame, last_obj, last_faces, last_emos)

            # FPS
            dt = time.perf_counter() - t0
            self.fps_buffer.append(1.0 / (dt + 1e-9))
            fps = np.mean(self.fps_buffer)

            draw_hud(output, fps, self.frame_count,
                     len(last_obj), len(last_faces))

            if writer:
                writer.write(output)

            cv2.imshow("Emotion-Aware Object Detection | Press Q to quit", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                path = f"results/screenshot_{self.screenshot_n:03d}.jpg"
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(path, output)
                print(f"[INFO] Screenshot saved → {path}")
                self.screenshot_n += 1

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\n[DONE] Processed {self.frame_count} frames. FPS avg: {np.mean(self.fps_buffer):.1f}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion-Aware Object Detection")
    parser.add_argument("--source", default=0,
                        help="0=webcam, or path to video file")
    parser.add_argument("--save",   action="store_true",
                        help="Save annotated output video")
    parser.add_argument("--output", default="results/output.avi",
                        help="Path for saved video")
    args = parser.parse_args()

    src = int(args.source) if str(args.source).isdigit() else args.source
    detector = EmotionAwareDetector(source=src,
                                     save_output=args.save,
                                     output_path=args.output)
    detector.run()
