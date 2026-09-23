import cv2
import mediapipe as mp
import numpy as np
import joblib
from keras_facenet import FaceNet
from datetime import datetime
import os
import csv
import time

# -----------------------------
# Configuration & constants
# -----------------------------
DATASET_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BTECH")
MODEL_PATH = "face_svm.pkl"
LABEL_MAP_PATH = "label_map.pkl"
THRESHOLD = 0.5
PERSISTENCE_THRESHOLD = 2  # seconds

# -----------------------------
# Helper functions
# -----------------------------
def ensure_exists(path, desc):
    """Exit if path doesn't exist."""
    if not os.path.exists(path):
        print(f"❌ {desc} not found: {path}")
        exit()

def get_embedding(img):
    """Return FaceNet embedding for given face image."""
    img = cv2.resize(img, (160, 160))
    img = np.expand_dims(img, axis=0)
    embedding = embedder.embeddings(img)
    return embedding[0]

# -----------------------------
# User input
# -----------------------------
print("Available classes: AI, EXTC, CIVIL, BSDS")
selected_class = input("Enter class: ").strip().upper()

print("Available years: 1, 2, 3, 4")
selected_year = input("Enter year: ").strip()

# -----------------------------
# Validate paths
# -----------------------------
CSV_STUDENT_PATH = os.path.join(DATASET_BASE_PATH, selected_class, selected_year, "faces")
ensure_exists(CSV_STUDENT_PATH, "Class/year folder")

all_students = [
    name for name in os.listdir(CSV_STUDENT_PATH)
    if os.path.isdir(os.path.join(CSV_STUDENT_PATH, name))
]
if not all_students:
    print(f"❌ No students found in folder: {CSV_STUDENT_PATH}")
    exit()

ensure_exists(MODEL_PATH, "Trained SVM model")
ensure_exists(LABEL_MAP_PATH, "Label map")

# -----------------------------
# Load models
# -----------------------------
embedder = FaceNet()
clf = joblib.load(MODEL_PATH)
label_map = joblib.load(LABEL_MAP_PATH)

# -----------------------------
# Prepare attendance CSV
# -----------------------------
CSV_FILE = f"attendance_{selected_class}_{selected_year}.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time", "Status"])

# -----------------------------
# Setup camera & face detection
# -----------------------------
mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(0)
visible_faces = {}
detected_students = set()

# -----------------------------
# Recognition loop
# -----------------------------
try:
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            current_time = time.time()

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                    x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)

                    # Clip coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    face = rgb[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    # Get embedding and predict
                    embedding = get_embedding(face).reshape(1, -1)
                    probs = clf.predict_proba(embedding)[0]
                    pred_idx = np.argmax(probs)
                    prob = probs[pred_idx]
                    name = label_map[pred_idx] if prob >= THRESHOLD else "Unknown"

                    # Draw results
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, f"{name} ({prob:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                    )

                    # Attendance persistence
                    if name == "Unknown":
                        continue

                    if name not in visible_faces:
                        visible_faces[name] = current_time
                    elif current_time - visible_faces[name] >= PERSISTENCE_THRESHOLD:
                        detected_students.add(name)
                        del visible_faces[name]

            # Show window
            cv2.imshow("Face Recognition Attendance", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Record attendance
# -----------------------------
today_date = datetime.now().strftime("%Y-%m-%d")
time_now = datetime.now().strftime("%H:%M:%S")

with open(CSV_FILE, mode="a", newline="") as f:
    writer = csv.writer(f)
    for student in all_students:
        status = "Present" if student in detected_students else "Absent"
        time_str = time_now if status == "Present" else ""
        writer.writerow([student, today_date, time_str, status])

print(f"✅ Attendance saved to {CSV_FILE}")