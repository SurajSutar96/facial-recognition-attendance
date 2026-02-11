"""
Facial Recognition Attendance System - Using face_recognition (dlib)
Accuracy: 99.38% on LFW benchmark
"""

import os
import sqlite3
import json
import cv2
import numpy as np
import face_recognition
from datetime import datetime, date
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from io import BytesIO
import aiofiles
import traceback
import time

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("known_faces", exist_ok=True)
os.makedirs("database", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Facial Recognition Attendance System", version="5.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('database/attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        email TEXT,
        department TEXT,
        semester TEXT,
        face_encoding_path TEXT,
        registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        name TEXT NOT NULL,
        date DATE NOT NULL,
        time TIME NOT NULL,
        status TEXT DEFAULT 'Present',
        FOREIGN KEY (student_id) REFERENCES students (student_id)
    )
    ''')
    
    cursor.execute('''
    CREATE UNIQUE INDEX IF NOT EXISTS idx_student_date 
    ON attendance (student_id, date)
    ''')
    
    conn.commit()
    conn.close()

init_db()

# Database helper functions
def get_db_connection():
    conn = sqlite3.connect('database/attendance.db')
    conn.row_factory = sqlite3.Row
    return conn

def add_student(student_id: str, name: str, email: str, department: str, semester: str, face_path: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO students (student_id, name, email, department, semester, face_encoding_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (student_id, name, email, department, semester, face_path))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_all_students():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM students ORDER BY name')
    students = cursor.fetchall()
    conn.close()
    return students

def mark_attendance(student_id: str, name: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    today = date.today().isoformat()
    current_time = datetime.now().strftime("%H:%M:%S")
    
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO attendance (student_id, name, date, time, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (student_id, name, today, current_time, 'Present'))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error marking attendance: {e}")
        return False
    finally:
        conn.close()

def get_today_attendance():
    conn = get_db_connection()
    cursor = conn.cursor()
    today = date.today().isoformat()
    
    cursor.execute('''
        SELECT a.*, s.department, s.semester 
        FROM attendance a
        LEFT JOIN students s ON a.student_id = s.student_id
        WHERE a.date = ?
        ORDER BY a.time DESC
    ''', (today,))
    attendance = cursor.fetchall()
    conn.close()
    return attendance

def get_attendance_report(start_date: str, end_date: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            s.student_id,
            s.name,
            s.department,
            s.semester,
            COUNT(DISTINCT a.date) as total_days,
            SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) as present_days,
            SUM(CASE WHEN a.status = 'Absent' THEN 1 ELSE 0 END) as absent_days
        FROM students s
        LEFT JOIN attendance a ON s.student_id = a.student_id 
            AND a.date BETWEEN ? AND ?
        GROUP BY s.student_id, s.name, s.department, s.semester
        ORDER BY s.department, s.name
    ''', (start_date, end_date))
    
    report = []
    for row in cursor.fetchall():
        record = dict(row)
        total_days = record['total_days'] or 0
        present_days = record['present_days'] or 0
        
        if total_days > 0:
            attendance_percentage = (present_days / total_days) * 100
        else:
            attendance_percentage = 0
            
        record['attendance_percentage'] = round(attendance_percentage, 1)
        report.append(record)
    
    conn.close()
    return report

class FaceRecognitionEngine:
    """Face recognition engine using the face_recognition library (dlib-based).
    
    Uses a 128-dimensional face encoding from a deep neural network
    trained on ~3 million face images. Achieves 99.38% accuracy on the
    Labeled Faces in the Wild (LFW) benchmark.
    """
    
    def __init__(self):
        self.tolerance = 0.5  # face_recognition distance threshold (lower = more strict)
        self.encodings_cache = {}  # student_id -> 128-d numpy array
        self.names_cache = {}      # student_id -> name
        
        print("Initializing Face Recognition Engine (dlib deep learning model)...")
        print(f"Tolerance: {self.tolerance} (lower = more strict)")
        
        self.migrate_old_embeddings()
        self.load_encodings()
    
    def migrate_old_embeddings(self):
        """Migrate old LBPH/histogram embeddings to face_recognition 128-d encodings.
        
        Old embeddings have shape (288,) or (288,) (histogram-based),
        new embeddings have shape (128,) (face_recognition deep learning).
        If old-format .npy files are detected, we re-encode from the _face.jpg images.
        """
        if not os.path.exists("known_faces"):
            return
        
        migrated = 0
        for filename in os.listdir("known_faces"):
            if not filename.endswith('.npy'):
                continue
            
            filepath = os.path.join("known_faces", filename)
            try:
                data = np.load(filepath)
                if data.shape != (128,):
                    student_id = filename.replace('.npy', '')
                    face_img_path = os.path.join("known_faces", f"{student_id}_face.jpg")
                    
                    if os.path.exists(face_img_path):
                        print(f"  Migrating {student_id}: old shape {data.shape} -> re-encoding from face image...")
                        image = face_recognition.load_image_file(face_img_path)
                        encodings = face_recognition.face_encodings(image)
                        
                        if encodings:
                            np.save(filepath, encodings[0])
                            print(f"  [OK] Migrated {student_id} to 128-d encoding")
                            migrated += 1
                        else:
                            print(f"  [FAIL] Could not detect face in {face_img_path}, removing old embedding")
                            os.remove(filepath)
                    else:
                        print(f"  [FAIL] No face image for {student_id}, removing incompatible embedding")
                        os.remove(filepath)
            except Exception as e:
                print(f"  [FAIL] Error migrating {filename}: {e}")
        
        if migrated > 0:
            print(f"[OK] Migrated {migrated} embeddings to face_recognition format")
    
    def load_encodings(self):
        """Load all 128-d face encodings from disk into cache."""
        self.encodings_cache = {}
        self.names_cache = {}
        
        if not os.path.exists("known_faces"):
            print("No known_faces directory found")
            return
        
        for filename in os.listdir("known_faces"):
            if not filename.endswith('.npy'):
                continue
            
            student_id = filename.replace('.npy', '')
            try:
                encoding = np.load(os.path.join("known_faces", filename))
                
                # Verify it's a valid 128-d encoding
                if encoding.shape != (128,):
                    print(f"  [FAIL] Skipping {student_id}: invalid encoding shape {encoding.shape}")
                    continue
                
                # Get student name from database
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT name FROM students WHERE student_id = ?', (student_id,))
                student = cursor.fetchone()
                conn.close()
                
                name = student['name'] if student else student_id
                
                self.encodings_cache[student_id] = encoding
                self.names_cache[student_id] = name
                print(f"  [OK] Loaded encoding for {name} ({student_id})")
            except Exception as e:
                print(f"  [FAIL] Error loading encoding for {student_id}: {e}")
        
        print(f"Total face encodings loaded: {len(self.encodings_cache)}")
    
    # Keep old attribute name for backward compatibility with debug endpoints
    @property
    def embeddings_cache(self):
        return self.encodings_cache
    
    @property
    def threshold(self):
        return self.tolerance
    
    def extract_encoding(self, image_path: str):
        """Extract 128-d face encoding from an image file.
        
        Uses face_recognition library which uses dlib's deep learning model.
        Returns the encoding for the first face found, or None if no face detected.
        """
        try:
            print(f"Extracting face encoding from: {image_path}")
            
            # Load image using face_recognition (expects RGB)
            image = face_recognition.load_image_file(image_path)
            
            # Detect face locations first
            face_locations = face_recognition.face_locations(image, model="hog")
            
            if not face_locations:
                print("  No face detected in the image")
                # Try with CNN model as fallback (more accurate but slower)
                print("  Retrying with more sensitive detection...")
                face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2, model="hog")
                
                if not face_locations:
                    print("  [FAIL] Still no face detected")
                    return None
            
            print(f"  Detected {len(face_locations)} face(s)")
            
            # Get encodings for detected faces
            encodings = face_recognition.face_encodings(image, face_locations)
            
            if not encodings:
                print("  [FAIL] Could not compute face encoding")
                return None
            
            encoding = encodings[0]  # Use the first face
            print(f"  [OK] Extracted 128-d face encoding")
            return encoding
            
        except Exception as e:
            print(f"  [FAIL] Error extracting encoding: {e}")
            traceback.print_exc()
            return None
    
    # Backward compatibility aliases
    def extract_face_features(self, image_path: str):
        return self.extract_encoding(image_path)
    
    def extract_embedding(self, image_path: str):
        return self.extract_encoding(image_path)
    
    def save_embedding(self, image_path: str, student_id: str):
        """Extract and save 128-d face encoding for a student."""
        try:
            encoding = self.extract_encoding(image_path)
            if encoding is None:
                print(f"[FAIL] Could not extract encoding for {student_id}")
                return None
            
            # Save encoding as .npy
            encoding_path = f"known_faces/{student_id}.npy"
            np.save(encoding_path, encoding)
            
            # Save a cropped face image for reference/display
            try:
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    # Add margin
                    margin = 30
                    h, w = image.shape[:2]
                    top = max(0, top - margin)
                    right = min(w, right + margin)
                    bottom = min(h, bottom + margin)
                    left = max(0, left - margin)
                    
                    face_img = image[top:bottom, left:right]
                    if face_img.size > 0:
                        # Convert RGB to BGR for cv2.imwrite
                        face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                        face_img_path = f"known_faces/{student_id}_face.jpg"
                        cv2.imwrite(face_img_path, face_img_bgr)
            except Exception as e:
                print(f"Note: Could not save cropped face image: {e}")
            
            # Update cache
            self.encodings_cache[student_id] = encoding
            
            print(f"[OK] Saved encoding for {student_id}")
            return encoding_path
            
        except Exception as e:
            print(f"[FAIL] Error saving embedding: {e}")
            traceback.print_exc()
            return None
    
    def recognize_from_image(self, image_path: str):
        """Recognize a face from an image file.
        
        Returns (student_id, distance) if recognized, (None, None) otherwise.
        Uses face_recognition.compare_faces and face_recognition.face_distance
        for accurate matching.
        """
        try:
            print(f"\n=== Starting face recognition ===")
            
            # Extract encoding from the input image
            encoding = self.extract_encoding(image_path)
            if encoding is None:
                print("[FAIL] No face encoding extracted")
                return None, None
            
            if not self.encodings_cache:
                print("[FAIL] No registered faces to compare against")
                return None, None
            
            # Prepare known encodings
            known_ids = list(self.encodings_cache.keys())
            known_encodings = [self.encodings_cache[sid] for sid in known_ids]
            
            # Compare against all known faces
            distances = face_recognition.face_distance(known_encodings, encoding)
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=self.tolerance)
            
            # Log comparisons
            for i, sid in enumerate(known_ids):
                name = self.names_cache.get(sid, sid)
                match_str = "[OK] MATCH" if matches[i] else "[FAIL]"
                print(f"  {match_str} {name} ({sid}): distance = {distances[i]:.4f}")
            
            # Find the best match (lowest distance among matches)
            best_match = None
            best_distance = float('inf')
            
            for i, (is_match, dist) in enumerate(zip(matches, distances)):
                if is_match and dist < best_distance:
                    best_distance = dist
                    best_match = known_ids[i]
            
            if best_match:
                name = self.names_cache.get(best_match, best_match)
                print(f"[OK] Recognized: {name} (distance: {best_distance:.4f})")
                return best_match, best_distance
            else:
                print(f"[FAIL] No match found (tolerance: {self.tolerance})")
                return None, None
            
        except Exception as e:
            print(f"[FAIL] Error in recognition: {e}")
            traceback.print_exc()
            return None, None

# Create face recognition engine instance
try:
    face_engine = FaceRecognitionEngine()
except Exception as e:
    print(f"Warning: Could not initialize face recognition engine: {e}")
    traceback.print_exc()
    face_engine = None

# Add template globals so Jinja2 can call get_all_students() directly
templates.env.globals['get_all_students'] = get_all_students

# Routes - All routes remain exactly the same as before
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return RedirectResponse("/dashboard")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    today_attendance = get_today_attendance()
    total_students = len(get_all_students())
    present_today = len(today_attendance)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "dashboard",
        "page_title": "Dashboard",
        "today_attendance": today_attendance,
        "total_students": total_students,
        "present_today": present_today
    })

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "register",
        "page_title": "Register Student"
    })

@app.post("/register")
async def register_student(
    request: Request,
    student_id: str = Form(...),
    name: str = Form(...),
    email: str = Form(None),
    department: str = Form(...),
    semester: str = Form(...),
    face_image: UploadFile = File(...)
):
    if face_engine is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "page": "register",
            "page_title": "Register Student",
            "message": "Face recognition engine not initialized. Please check server logs.",
            "message_type": "error"
        })
    
    try:
        print(f"\n=== Starting registration for {student_id} - {name} ===")
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        file_ext = os.path.splitext(face_image.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "page": "register",
                "page_title": "Register Student",
                "message": "Please upload a valid image file (JPG, JPEG, PNG, BMP)",
                "message_type": "error"
            })
        
        # Create upload directory
        upload_dir = "static/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate safe filename
        safe_name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')[:50]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{student_id}_{safe_name}_{timestamp}{file_ext}"
        image_path = f"{upload_dir}/{filename}"
        
        print(f"Saving image to: {image_path}")
        
        # Save the uploaded file
        async with aiofiles.open(image_path, 'wb') as out_file:
            content = await face_image.read()
            await out_file.write(content)
        
        # Verify image was saved
        if not os.path.exists(image_path):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "page": "register",
                "page_title": "Register Student",
                "message": "Failed to save image file",
                "message_type": "error"
            })
        
        # Extract and save face encoding
        print("Extracting face encoding...")
        encoding_path = face_engine.save_embedding(image_path, student_id)
        
        if encoding_path:
            # Add student to database
            success = add_student(student_id, name, email, department, semester, encoding_path)
            
            if success:
                print(f"[OK] Successfully registered {name} ({student_id})")
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "page": "register",
                    "page_title": "Register Student",
                    "message": f"Student {name} registered successfully!",
                    "message_type": "success"
                })
            else:
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "page": "register",
                    "page_title": "Register Student",
                    "message": "Student ID already exists!",
                    "message_type": "error"
                })
        else:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "page": "register",
                "page_title": "Register Student",
                "message": "Could not detect face in the image. Please upload a clear frontal face photo with good lighting.",
                "message_type": "error"
            })
            
    except Exception as e:
        print(f"[FAIL] Registration error: {e}")
        traceback.print_exc()
        return templates.TemplateResponse("index.html", {
            "request": request,
            "page": "register",
            "page_title": "Register Student",
            "message": f"Registration failed: {str(e)}",
            "message_type": "error"
        })

@app.get("/recognize", response_class=HTMLResponse)
async def recognize_page(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "recognize",
        "page_title": "Take Attendance"
    })

@app.post("/api/recognize")
async def api_recognize(file: UploadFile = File(...)):
    """API endpoint for face recognition"""
    if face_engine is None:
        return JSONResponse({
            "recognized": False,
            "error": "Face recognition engine not initialized"
        }, status_code=500)
    
    temp_path = None
    try:
        print("\n=== Face recognition request received ===")
        
        # Save uploaded image temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_path = f"temp_capture_{timestamp}.jpg"
        
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        print(f"Saved temporary image: {temp_path}")
        
        # Check if image exists
        if not os.path.exists(temp_path):
            return JSONResponse({
                "recognized": False,
                "error": "Failed to save temporary image"
            })
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size < 1024:  # Less than 1KB
            return JSONResponse({
                "recognized": False,
                "error": "Image file is too small"
            })
        
        # Recognize face
        student_id, distance = face_engine.recognize_from_image(temp_path)
        
        if student_id:
            # Get student info from database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM students WHERE student_id = ?', (student_id,))
            student = cursor.fetchone()
            conn.close()
            
            if student:
                name = student['name']
                # Mark attendance
                success = mark_attendance(student_id, name)
                
                return JSONResponse({
                    "recognized": True,
                    "student": {
                        "student_id": student_id,
                        "name": name
                    },
                    "distance": float(distance) if distance else 0.0,
                    "attendance_marked": success,
                    "message": f"Welcome {name}! Attendance marked."
                })
        
        return JSONResponse({
            "recognized": False,
            "message": "Face not recognized. Make sure: 1) You are registered 2) Face is clearly visible 3) Good lighting"
        })
        
    except Exception as e:
        print(f"[FAIL] Recognition API error: {e}")
        traceback.print_exc()
        return JSONResponse({
            "recognized": False,
            "error": f"Recognition failed: {str(e)}"
        }, status_code=500)
        
    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Cleaned up temp file: {temp_path}")
            except:
                pass

@app.get("/attendance", response_class=HTMLResponse)
async def attendance_page(request: Request):
    today_attendance = get_today_attendance()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "attendance",
        "page_title": "Today's Attendance",
        "attendance": today_attendance
    })

@app.get("/report", response_class=HTMLResponse)
async def report_page(request: Request):
    today = date.today()
    first_day = today.replace(day=1).isoformat()
    
    report = get_attendance_report(first_day, today.isoformat())
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "report",
        "page_title": "Attendance Report",
        "report": report,
        "start_date": first_day,
        "end_date": today.isoformat()
    })

@app.post("/report")
async def generate_report(
    request: Request,
    start_date: str = Form(...),
    end_date: str = Form(...)
):
    report = get_attendance_report(start_date, end_date)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "report",
        "page_title": "Attendance Report",
        "report": report,
        "start_date": start_date,
        "end_date": end_date
    })

@app.get("/students", response_class=HTMLResponse)
async def students_page(request: Request):
    students = get_all_students()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "page": "students",
        "page_title": "All Students",
        "students": students
    })

@app.get("/api/attendance/today")
async def get_today_attendance_api():
    attendance = get_today_attendance()
    return JSONResponse([
        {
            "student_id": record['student_id'],
            "name": record['name'],
            "time": record['time'],
            "department": record['department'] if 'department' in record else '',
            "status": record['status']
        }
        for record in attendance
    ])

@app.get("/api/attendance/stats")
async def get_attendance_stats():
    today = date.today().isoformat()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) as total FROM students')
    total_students = cursor.fetchone()['total']
    
    cursor.execute('SELECT COUNT(DISTINCT student_id) as present FROM attendance WHERE date = ?', (today,))
    present_today = cursor.fetchone()['present']
    
    conn.close()
    
    return JSONResponse({
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": total_students - present_today
    })

@app.post("/api/attendance/mark")
async def mark_attendance_api(request: Request):
    """API endpoint to mark attendance manually"""
    try:
        data = await request.json()
        student_id = data.get('student_id')
        name = data.get('name')
        
        if not student_id or not name:
            raise HTTPException(status_code=400, detail="Missing student information")
        
        success = mark_attendance(student_id, name)
        
        return JSONResponse({
            "success": success,
            "message": "Attendance marked successfully" if success else "Attendance already marked"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/faces")
async def debug_faces():
    """Debug endpoint to check registered faces"""
    try:
        known_faces = []
        
        # Check embeddings
        if os.path.exists("known_faces"):
            for filename in os.listdir("known_faces"):
                if filename.endswith('.npy'):
                    student_id = filename.replace('.npy', '')
                    file_path = f"known_faces/{filename}"
                    
                    try:
                        embedding = np.load(file_path)
                        file_size = os.path.getsize(file_path)
                        
                        # Get student info
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute('SELECT name FROM students WHERE student_id = ?', (student_id,))
                        student = cursor.fetchone()
                        conn.close()
                        
                        known_faces.append({
                            "student_id": student_id,
                            "name": student['name'] if student else "Unknown",
                            "embedding_shape": str(embedding.shape),
                            "file_size_kb": file_size // 1024,
                            "has_face_image": os.path.exists(f"known_faces/{student_id}_face.jpg")
                        })
                    except Exception as e:
                        known_faces.append({
                            "student_id": student_id,
                            "error": str(e)
                        })
        
        return JSONResponse({
            "status": "ok",
            "total_embeddings": len(known_faces),
            "embeddings_in_cache": len(face_engine.embeddings_cache) if face_engine else 0,
            "recognition_threshold": face_engine.threshold if face_engine else 0,
            "engine_status": "initialized" if face_engine else "not initialized",
            "faces": known_faces
        })
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@app.get("/api/face-test")
async def face_test():
    """Test endpoint to verify face recognition works"""
    try:
        if face_engine is None:
            return JSONResponse({
                "status": "error",
                "error": "Face recognition engine not initialized"
            }, status_code=500)
        
        # Test embeddings
        test_embeddings = []
        if os.path.exists("known_faces"):
            npy_files = [f for f in os.listdir("known_faces") if f.endswith('.npy')]
            for npy_file in npy_files[:3]:  # Test first 3
                student_id = npy_file.replace('.npy', '')
                embedding_path = f"known_faces/{npy_file}"
                try:
                    embedding = np.load(embedding_path)
                    test_embeddings.append({
                        "student_id": student_id,
                        "shape": str(embedding.shape),
                        "loaded": True
                    })
                except Exception as e:
                    test_embeddings.append({
                        "student_id": student_id,
                        "error": str(e)
                    })
        
        return JSONResponse({
            "status": "ready",
            "engine_loaded": face_engine is not None,
            "embeddings_count": len(face_engine.embeddings_cache) if face_engine else 0,
            "test_embeddings": test_embeddings,
            "message": f"Face recognition system ready with {len(face_engine.embeddings_cache) if face_engine else 0} registered faces" if face_engine else "Face engine not loaded"
        })
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

@app.post("/api/test-face-detection")
async def test_face_detection(file: UploadFile = File(...)):
    """Test face detection without recognition"""
    temp_path = None
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_path = f"temp_test_{timestamp}.jpg"
        
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Test face detection using face_recognition library
        image = face_recognition.load_image_file(temp_path)
        face_locations = face_recognition.face_locations(image, model="hog")
        
        return JSONResponse({
            "faces_detected": len(face_locations),
            "detection_details": [
                {
                    "top": int(top),
                    "right": int(right),
                    "bottom": int(bottom),
                    "left": int(left)
                }
                for (top, right, bottom, left) in face_locations
            ],
            "message": f"Detected {len(face_locations)} face(s) in the image"
        })
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "faces_detected": 0
        }, status_code=500)
        
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.delete("/api/students/{student_id}")
async def delete_student(student_id: str):
    """Delete a registered student and their face data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if student exists
        cursor.execute('SELECT name FROM students WHERE student_id = ?', (student_id,))
        student = cursor.fetchone()
        
        if not student:
            conn.close()
            raise HTTPException(status_code=404, detail="Student not found")
        
        student_name = student['name']
        
        # Delete from database
        cursor.execute('DELETE FROM attendance WHERE student_id = ?', (student_id,))
        cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
        conn.commit()
        conn.close()
        
        # Delete face encoding files
        npy_path = f"known_faces/{student_id}.npy"
        face_path = f"known_faces/{student_id}_face.jpg"
        
        if os.path.exists(npy_path):
            os.remove(npy_path)
        if os.path.exists(face_path):
            os.remove(face_path)
        
        # Remove from cache
        if face_engine:
            face_engine.encodings_cache.pop(student_id, None)
            face_engine.names_cache.pop(student_id, None)
        
        return JSONResponse({
            "success": True,
            "message": f"Student {student_name} ({student_id}) deleted successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/students/{student_id}")
async def get_student_details(student_id: str):
    """Get details for a specific student"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
        student = cursor.fetchone()
        
        if not student:
            conn.close()
            raise HTTPException(status_code=404, detail="Student not found")
        
        # Get attendance stats
        cursor.execute('''
            SELECT COUNT(*) as total_days,
                   SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) as present_days
            FROM attendance WHERE student_id = ?
        ''', (student_id,))
        stats = cursor.fetchone()
        conn.close()
        
        total_days = stats['total_days'] or 0
        present_days = stats['present_days'] or 0
        attendance_pct = round((present_days / total_days * 100), 1) if total_days > 0 else 0
        
        return JSONResponse({
            "student_id": student['student_id'],
            "name": student['name'],
            "email": student['email'] or 'N/A',
            "department": student['department'] or 'N/A',
            "semester": student['semester'] or 'N/A',
            "registration_date": student['registration_date'],
            "total_attendance_days": total_days,
            "present_days": present_days,
            "attendance_percentage": attendance_pct,
            "has_face_encoding": os.path.exists(f"known_faces/{student_id}.npy")
        })
        

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/students")
async def list_students_api():
    """List all registered students as JSON (used by script.js)"""
    students = get_all_students()
    return JSONResponse([
        {
            "student_id": s['student_id'],
            "name": s['name'],
            "email": s['email'] if s['email'] else 'N/A',
            "department": s['department'] if s['department'] else 'N/A',
            "semester": s['semester'] if s['semester'] else 'N/A',
            "registration_date": s['registration_date'] if s['registration_date'] else 'N/A'
        }
        for s in students
    ])

@app.post("/api/export/today")
async def export_today_attendance():
    """Export today's attendance as Excel file"""
    try:
        attendance = get_today_attendance()
        if not attendance:
            return JSONResponse({"error": "No attendance records for today"}, status_code=404)
        
        # Create DataFrame
        data = [{
            "Student ID": r['student_id'],
            "Name": r['name'],
            "Time": r['time'],
            "Status": r['status'],
            "Department": r['department'] if 'department' in r else 'N/A',
            "Semester": r['semester'] if 'semester' in r else 'N/A'
        } for r in attendance]
        
        df = pd.DataFrame(data)
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Today_Attendance')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Today_Attendance']
            for idx, col in enumerate(df.columns):
                max_len = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = max_len
        
        output.seek(0)
        
        today_str = date.today().strftime("%Y-%m-%d")
        headers = {
            'Content-Disposition': f'attachment; filename="attendance_{today_str}.xlsx"'
        }
        
        return StreamingResponse(
            output,
            headers=headers,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
            
    except Exception as e:
        print(f"Export error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/export/attendance")
async def export_attendance_report_api(request: Request):
    """Export attendance for a date range as Excel"""
    try:
        data = await request.json()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        if not start_date or not end_date:
            return JSONResponse({"error": "Start date and end date are required"}, status_code=400)
            
        report = get_attendance_report(start_date, end_date)
        
        if not report:
            return JSONResponse({"error": "No records found for the selected period"}, status_code=404)
        
        # Create DataFrame
        df = pd.DataFrame(report)
        
        # Rename columns for better readability
        df = df.rename(columns={
            'student_id': 'Student ID',
            'name': 'Name',
            'department': 'Department',
            'semester': 'Semester',
            'total_days': 'Total Days',
            'present_days': 'Present Days',
            'absent_days': 'Absent Days',
            'attendance_percentage': 'Attendance %'
        })
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance_Report')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Attendance_Report']
            for idx, col in enumerate(df.columns):
                max_len = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = max_len
                
        output.seek(0)
        
        headers = {
            'Content-Disposition': f'attachment; filename="attendance_report_{start_date}_to_{end_date}.xlsx"'
        }
        
        return StreamingResponse(
            output,
            headers=headers,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        print(f"Export error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Facial Recognition Attendance System")
    print("Engine: face_recognition (dlib deep learning)")
    print("Accuracy: 99.38% on LFW benchmark")
    print("=" * 60)
    if face_engine:
        print(f"Registered faces: {len(face_engine.encodings_cache)}")
        print(f"Recognition tolerance: {face_engine.tolerance}")
    else:
        print("WARNING: Face recognition engine failed to initialize!")
    print("Visit: http://localhost:8000")
    print("Debug: http://localhost:8000/debug/faces")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)