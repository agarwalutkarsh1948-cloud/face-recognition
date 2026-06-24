"""
FaceAttend - Smart Attendance System
Flask Backend with OpenCV Face Recognition
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import os
import csv
import json
from datetime import datetime, date
import threading
import base64

app = Flask(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR    = "dataset"          # har student ki photos yahan hain
ATTENDANCE_DIR = "attendance_logs"  # CSV files yahan save hongi
MODEL_FILE     = "models/trainer.yml"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ─── Global state ─────────────────────────────────────────────────────────────
camera        = None
camera_lock   = threading.Lock()
recognizer    = cv2.face.LBPHFaceRecognizer_create()
face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
label_map     = {}   # id → { name, roll, department }
model_trained = False

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_label_map():
    """students.json se label map load karo"""
    global label_map
    path = "models/students.json"
    if os.path.exists(path):
        with open(path) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}

def save_label_map():
    with open("models/students.json", "w") as f:
        json.dump(label_map, f, indent=2)

def load_model():
    """Agar trained model hai toh load karo"""
    global model_trained
    if os.path.exists(MODEL_FILE):
        recognizer.read(MODEL_FILE)
        load_label_map()
        model_trained = True

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def release_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None

def get_today_csv():
    """Aaj ki attendance CSV ka path"""
    today = date.today().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")

def already_marked(roll):
    """Check karo agar student ne aaj attendance mark ki hai"""
    csv_path = get_today_csv()
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("roll") == roll:
                return True
    return False

def save_attendance(name, roll, department, confidence):
    csv_path = get_today_csv()
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name","roll","department","time","confidence","date"])
        if write_header:
            w.writeheader()
        w.writerow({
            "name":       name,
            "roll":       roll,
            "department": department,
            "time":       datetime.now().strftime("%I:%M %p"),
            "confidence": f"{confidence:.1f}%",
            "date":       date.today().strftime("%d-%m-%Y"),
        })

# ─── Frame generator (MJPEG stream) ──────────────────────────────────────────

def generate_frames():
    while True:
        with camera_lock:
            cam = get_camera()
            success, frame = cam.read()
        if not success:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (200, 200))

            if model_trained and len(label_map) > 0:
                label_id, raw_dist = recognizer.predict(roi_resized)
                # LBPH: distance chhoti = acha match. 0-100 range mein convert
                confidence = max(0, 100 - raw_dist)
                student = label_map.get(label_id, {})
                name    = student.get("name", "Unknown")

                if confidence >= 45:
                    color = (34, 180, 100)   # green → match
                    text  = f"{name}  {confidence:.0f}%"
                else:
                    color = (60, 100, 220)   # blue → low confidence
                    text  = f"Unknown  {confidence:.0f}%"
            else:
                color = (180, 180, 60)
                text  = "Model nahi hai"

            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.rectangle(frame, (x, y-28), (x+w, y), color, -1)
            cv2.putText(frame, text, (x+6, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        if len(faces) == 0 and model_trained:
            cv2.putText(frame, "Chehra nahi mila...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 180, 255), 2)

        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")

@app.route("/admin")
def admin_page():
    return render_template("admin.html")

# Live camera stream
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ─── API: Student register karo + photos capture karo ────────────────────────

@app.route("/api/register", methods=["POST"])
def api_register():
    """
    Naye student ko register karo.
    Body: { name, roll, department }
    Pehle photos capture karni hongi /api/capture se.
    """
    data = request.json
    name       = data.get("name", "").strip()
    roll       = data.get("roll", "").strip().upper()
    department = data.get("department", "").strip()

    if not all([name, roll, department]):
        return jsonify({"success": False, "message": "Saari fields zaroori hain"}), 400

    # Check duplicate
    for v in label_map.values():
        if v.get("roll") == roll:
            return jsonify({"success": False, "message": f"{roll} already registered hai"}), 409

    # New label ID assign karo
    label_id = max(label_map.keys(), default=-1) + 1
    label_map[label_id] = {"name": name, "roll": roll, "department": department}
    save_label_map()

    student_dir = os.path.join(DATASET_DIR, roll)
    os.makedirs(student_dir, exist_ok=True)

    return jsonify({"success": True, "label_id": label_id,
                    "message": f"{name} register ho gaya! Ab photos capture karo."})

@app.route("/api/capture", methods=["POST"])
def api_capture():
    """
    Webcam se ek frame lo aur student ki photo save karo.
    Body: { roll, photo_count }   (photo_count = kitni photos ab tak li hain)
    """
    data        = request.json
    roll        = data.get("roll", "").strip().upper()
    photo_count = int(data.get("photo_count", 0))

    student_dir = os.path.join(DATASET_DIR, roll)
    if not os.path.exists(student_dir):
        return jsonify({"success": False, "message": "Pehle register karo"}), 400

    with camera_lock:
        cam = get_camera()
        success, frame = cam.read()

    if not success:
        return jsonify({"success": False, "message": "Camera se frame nahi mili"}), 500

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))

    if len(faces) == 0:
        return jsonify({"success": False, "message": "Koi chehra nahi mila, seedha dekhein"}), 400

    x, y, w, h = faces[0]
    roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

    filename = os.path.join(student_dir, f"{photo_count+1}.jpg")
    cv2.imwrite(filename, roi)

    return jsonify({"success": True, "saved": photo_count + 1,
                    "message": f"Photo {photo_count+1} save ho gayi"})

# ─── API: Model train karo ────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def api_train():
    """
    Dataset/  ke saare students ki photos se LBPH model train karo.
    """
    global model_trained
    faces_data = []
    labels     = []

    for label_id, student in label_map.items():
        roll        = student["roll"]
        student_dir = os.path.join(DATASET_DIR, roll)
        if not os.path.exists(student_dir):
            continue
        for fname in os.listdir(student_dir):
            if fname.endswith(".jpg"):
                img_path = os.path.join(student_dir, fname)
                img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces_data.append(img)
                    labels.append(label_id)

    if len(faces_data) < 2:
        return jsonify({"success": False,
                        "message": "Kam se kam 1 student ke 2 photos chahiye"}), 400

    recognizer.train(faces_data, np.array(labels))
    recognizer.save(MODEL_FILE)
    model_trained = True

    return jsonify({"success": True,
                    "message": f"Model train ho gaya! {len(set(labels))} students, {len(faces_data)} photos."})

# ─── API: Attendance mark karo ───────────────────────────────────────────────

@app.route("/api/mark_attendance", methods=["POST"])
def api_mark_attendance():
    """
    Camera se face lo, recognize karo, aur attendance CSV mein save karo.
    """
    if not model_trained:
        return jsonify({"success": False, "message": "Pehle model train karo"}), 400

    with camera_lock:
        cam = get_camera()
        success, frame = cam.read()

    if not success:
        return jsonify({"success": False, "message": "Camera se frame nahi mili"}), 500

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))

    if len(faces) == 0:
        return jsonify({"success": False, "message": "Chehra nahi mila, seedha dekhein"}), 400

    x, y, w, h = faces[0]
    roi        = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    label_id, raw_dist = recognizer.predict(roi)
    confidence = max(0, 100 - raw_dist)

    if confidence < 45:
        return jsonify({"success": False,
                        "message": f"Pehchaan nahi hua ({confidence:.0f}% confidence). Phir try karo."}), 401

    student = label_map.get(label_id)
    if not student:
        return jsonify({"success": False, "message": "Student record nahi mila"}), 404

    name       = student["name"]
    roll       = student["roll"]
    department = student["department"]

    if already_marked(roll):
        return jsonify({"success": False, "already": True,
                        "message": f"{name}, aapki attendance aaj pehle se mark hai!",
                        "name": name, "roll": roll})

    save_attendance(name, roll, department, confidence)

    return jsonify({"success": True, "name": name, "roll": roll,
                    "department": department, "confidence": round(confidence, 1),
                    "time": datetime.now().strftime("%I:%M %p"),
                    "message": f"Attendance mark ho gayi! Welcome, {name}"})

# ─── API: Aaj ki attendance list ─────────────────────────────────────────────

@app.route("/api/today_attendance")
def api_today_attendance():
    csv_path = get_today_csv()
    records  = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            records = list(csv.DictReader(f))
    return jsonify({"success": True, "records": records,
                    "total": len(records), "date": date.today().strftime("%d-%m-%Y")})

@app.route("/api/students")
def api_students():
    return jsonify({"success": True, "students": label_map,
                    "total": len(label_map)})

@app.route("/api/status")
def api_status():
    return jsonify({
        "model_trained": model_trained,
        "students":      len(label_map),
        "camera_ok":     True,
    })

# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    print("\n✅  FaceAttend server chal raha hai → http://127.0.0.1:5000\n")
    app.run(debug=True, threaded=True)
