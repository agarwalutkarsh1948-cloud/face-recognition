# FaceAttend – Smart Attendance System
### Flask + OpenCV Face Recognition Backend

---

## Project Structure

```
face_attendance/
├── app.py                  ← Main Flask server (sabse important file)
├── requirements.txt        ← Dependencies
├── dataset/                ← Students ki captured photos
│   └── 21CS047/            ← Roll number ke naam se folder
│       ├── 1.jpg
│       └── ...30.jpg
├── models/
│   ├── trainer.yml         ← Trained LBPH model (auto-generate hota hai)
│   └── students.json       ← Student label map
├── attendance_logs/
│   └── attendance_2025-06-25.csv  ← Har din ki CSV
├── static/css/style.css
└── templates/
    ├── base.html
    ├── index.html
    ├── attendance.html
    ├── register.html
    └── admin.html
```

---

## Setup (First Time)

### Step 1 – Python environment banao
```bash
cd face_attendance
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 2 – Dependencies install karo
```bash
pip install -r requirements.txt
```

> ⚠️  `opencv-contrib-python` zaroori hai – sirf `opencv-python` se LBPH nahi chalega.

### Step 3 – Server start karo
```bash
python app.py
```

Browser mein kholo: **http://127.0.0.1:5000**

---

## Use Kaise Karo (Step-by-Step)

### 1️⃣  Naya Student Register Karo
- `/register` page pe jao
- Naam, Roll Number, Department bharo
- "Register Karo" dabao
- Camera ke saamne baithein aur **30 baar "Capture" dabao**
- Phir **"Model Train Karo"** dabao

### 2️⃣  Attendance Mark Karo
- `/attendance` page pe jao
- Camera ke saamne baithein
- **"Attendance Mark Karo"** dabao
- System automatically chehra pehchaan kar attendance save karega

### 3️⃣  Admin Dashboard
- `/admin` pe saare records dekhein
- Students list, aaj ki attendance, model status

---

## API Endpoints

| Method | URL | Kaam |
|--------|-----|------|
| GET  | `/video_feed`           | Live MJPEG camera stream |
| POST | `/api/register`         | Naya student register    |
| POST | `/api/capture`          | Ek photo capture karo    |
| POST | `/api/train`            | Model train karo         |
| POST | `/api/mark_attendance`  | Attendance mark karo     |
| GET  | `/api/today_attendance` | Aaj ki attendance list   |
| GET  | `/api/students`         | Saare students           |
| GET  | `/api/status`           | System status            |

---

## Attendance CSV Format

```csv
name,roll,department,time,confidence,date
Priya Sharma,21CS047,Computer Science (CS),09:07 AM,98.4%,25-06-2025
Rahul Tiwari,21CS032,Computer Science (CS),09:11 AM,91.2%,25-06-2025
```

---

## Common Errors

| Error | Solution |
|-------|----------|
| `cv2.face` not found | `pip install opencv-contrib-python` |
| Camera nahi mili | Doosra USB port try karo, ya `VideoCapture(1)` |
| Low confidence | Aur photos lo (50+ behtar hai), achhi roshni mein |
| Model train nahi hua | Pehle kam se kam 1 student register karo |

---

## Tech Stack

- **Backend**: Python 3.10+ · Flask 2.3+
- **Face Detection**: OpenCV Haar Cascade
- **Face Recognition**: LBPH (Local Binary Pattern Histogram)
- **Storage**: CSV files (MySQL se replace kar sakte hain)
- **Frontend**: Jinja2 templates + Vanilla JS

---

*Made with ❤️ for FaceAttend project*
