"""
Evaluation Script — Emotion-Aware Object Detection
Computes: Accuracy, Precision, Recall, F1, Confusion Matrix
for emotion classification on a labelled image folder.

Folder structure expected:
  eval_data/
    happy/    *.jpg
    sad/      *.jpg
    angry/    *.jpg
    ...

Usage:
  python evaluate.py --data_dir eval_data/
"""

import os, argparse, time
import numpy as np
import cv2
from collections import defaultdict

try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False
    print("[WARN] DeepFace not installed — using random predictions for demo.")

EMOTIONS = ["happy","sad","angry","surprise","fear","disgust","neutral"]


def predict_emotion(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    if DEEPFACE_OK:
        try:
            res = DeepFace.analyze(img, actions=["emotion"],
                                   enforce_detection=False, silent=True)
            return (res[0] if isinstance(res, list) else res)["dominant_emotion"]
        except Exception:
            pass
    import random
    return random.choice(EMOTIONS)


def evaluate(data_dir):
    y_true, y_pred = [], []
    class_counts = defaultdict(int)

    print(f"\n{'─'*50}")
    print(f"  Evaluating on: {data_dir}")
    print(f"{'─'*50}")

    for label in EMOTIONS:
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".png",".jpeg"))]
        print(f"  {label:10s}: {len(files):4d} images")
        for fname in files:
            pred = predict_emotion(os.path.join(folder, fname))
            if pred is None:
                continue
            y_true.append(label)
            y_pred.append(pred)
            class_counts[label] += 1

    if not y_true:
        print("[ERROR] No images found. Check folder structure.")
        return

    labels = sorted(set(y_true))
    n = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}

    # Confusion matrix
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_idx and p in label_idx:
            cm[label_idx[t]][label_idx[p]] += 1

    # Per-class metrics
    print(f"\n{'─'*60}")
    print(f"  {'Emotion':<12} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"{'─'*60}")

    precs, recs, f1s = [], [], []
    for i, lbl in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        precs.append(prec); recs.append(rec); f1s.append(f1)
        print(f"  {lbl:<12} {prec:6.3f} {rec:6.3f} {f1:6.3f} {class_counts[lbl]:>8}")

    acc = sum(t==p for t,p in zip(y_true,y_pred)) / len(y_true)
    print(f"{'─'*60}")
    print(f"  {'OVERALL':<12} {np.mean(precs):6.3f} {np.mean(recs):6.3f} "
          f"{np.mean(f1s):6.3f} {len(y_true):>8}")
    print(f"\n  Accuracy : {acc*100:.2f}%")
    print(f"  macro-F1 : {np.mean(f1s)*100:.2f}%")
    print(f"{'─'*60}\n")

    # Print confusion matrix
    print("  Confusion Matrix (rows=true, cols=pred):")
    header = "  " + "".join(f"{l[:5]:>6}" for l in labels)
    print(header)
    for i, lbl in enumerate(labels):
        row = "  " + f"{lbl[:5]:<5}" + "".join(f"{cm[i,j]:>6}" for j in range(n))
        print(row)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="eval_data",
                        help="Root folder with sub-folders per emotion class")
    args = parser.parse_args()
    evaluate(args.data_dir)
