# Facial Recognition Attendance System

A robust, AI-powered attendance management system that uses facial recognition technology to automate the attendance marking process defined for educational institutions (Final Year Project).

## 📌 Project Overview

This system streamlines the traditional attendance marking process by using computer vision and deep learning. It detects and recognizes faces in real-time to mark attendance automatically, preventing proxy attendance and saving valuable class time. The application provides a user-friendly dashboard for students and administrators to manage and view attendance records.

## 🚀 Key Features

- **Real-time Face Recognition:** Uses state-of-the-art deep learning models (`dlib`, `face_recognition`) for high-accuracy face detection and recognition (99.38% accuracy).
- **Automatic Attendance Marking:** Instantly marks attendance when a registered student is recognized.
- **Proxy Prevention:** Ensures that only the actual student is present.
- **Student Registration:** Easy registration process with photo capture and personal details.
- **Admin Dashboard:** Comprehensive dashboard to view real-time statistics (Present/Absent/Total).
- **Report Generation:** Generate detailed attendance reports for specific date ranges.
- **Export Functionality:** Export daily attendance sheets and monthly reports to Excel (`.xlsx`).
- **Modern UI:** A clean, responsive user interface built with HTML5, CSS3, and JavaScript.

## 🛠️ Technology Stack

- **Backend:** Python 3.x, FastAPI (High-performance web framework)
- **Computer Vision:** OpenCV, face_recognition, dlib
- **Database:** SQLite (Lightweight, serverless database)
- **Frontend:** HTML5, CSS3, JavaScript
- **Data Processing:** Pandas, NumPy
- **Templating:** Jinja2

## 📋 Prerequisites

- Python 3.8 or higher
- CMake (Required for building `dlib`)
- Visual Studio C++ Build Tools (Windows only, for compiling dlib)

## ⚙️ Installation & Setup

1.  **Clone the Repository**

    ```bash
    git clone <repository-url>
    cd Attendance
    ```

2.  **Create a Virtual Environment**

    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Linux/Mac
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**

    ```bash
    python main.py
    ```

    OR

    ```bash
    uvicorn main:app --reload
    ```

5.  **Access the Dashboard**
    Open your web browser and navigate to: `http://localhost:8000`

## 📖 Usage Guide

1.  **Dashboard:** The home screen shows today's summary (Present, Absent, Total).
2.  **Register Student:** Go to the "Register" page. Enter student details (ID, Name, Dept, Semester) and upload or capture a clear frontal face photo.
3.  **Take Attendance:** Go to "Take Attendance". The camera will start. Students should look at the camera. Using the "Start Camera" button, the system will detect and recognize faces. Once recognized, attendance is marked automatically.
4.  **View Reports:** Go to "Attendance Report". Select a start and end date to generate a detailed report of student attendance.
5.  **Export Data:** Use the "Export" buttons in "Today's Attendance" or "Attendance Report" to download Excel sheets.

## 🔮 Future Enhancements

- Integration with external IP cameras for classroom surveillance.
- Mobile app integration for students to view their own attendance.
- SMS/Email notification to parents for absentee students.
- Cloud database integration (PostgreSQL/MySQL) for scalability.

## 👨‍💻 Developed By

- **[Your Name]** (Final Year Project)
- **Department of Computer Science**

---

© 2026 Facial Recognition Attendance System. All rights reserved.
