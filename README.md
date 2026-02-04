# VestaFace V2.35 - Deployment Guide

VestaFace is an automated reception system combining **ArcFace** facial recognition with the **Vestaboard Local API**.

## 1. Prerequisites
* **Python 3.10 - 3.12**: Do not use 3.13 due to library compatibility.
* **C++ Build Tools**: Microsoft Visual C++ 14.0+ is required for `insightface`.
  * Download and install Visual Studio 2026 Community Installer from https://visualstudio.microsoft.com/downloads/
  * Run the Visual Studio Community installer and select "Desktop development with C++."
* **Python Libraries**: Install via pip:
  * `python3 -m pip3 install --upgrade pip`
  * `pip3 install cv2-python numpy requests insightface onnxruntime tkinter`

## 2. Configuration Files
The script looks for two files in the same folder as the script:
* **`vb.url`**: Plain text file containing your Local API URL (e.g., `http://192.168.6.97:7000/local-api/message`).
* **`vb.key`**: Plain text file containing your Local API Key.

## 3. Usage
* **Camera Selection**: On startup, click the camera preview you wish to use.
* **Enrollment**: Unknown faces trigger a "NEW VISITOR:" prompt. Type name and press **Enter**.
* **Self-Correction**: If a face is matched while you are typing, the prompt cancels automatically.
* **Watchdog**: If a prompt is ignored for 30 seconds, it will auto-cancel to keep scanning.
* **Admin Dashboard**: Click the **ADMIN** button in the top-left corner to manage users, view session logs, and run backups.

## 4. File Management
* **`face_db.npz`**: Stores all facial data. Updates instantly on enrollment.
* **`visitor_log.txt`**: A persistent history of all visits and enrollments.
* **`backups/`**: Folder created automatically for manual data archives.