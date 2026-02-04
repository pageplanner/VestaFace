import os
import sys

# Silence OpenCV and Hardware backends
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["VIDEOIO_PRIORITY_MSMF"] = "0" 

import face_recognition
import cv2
import pickle
import tkinter as tk
from tkinter import simpledialog
import numpy as np
import time
import json
import requests
import threading
from datetime import datetime

# --- HELPERS ---

def get_timestamp():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

# --- VESTABOARD LOGIC ---

def load_vestaboard_config():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'vb.url'), 'r') as f:
        vestaboard_url = f.read().strip()
    with open(os.path.join(dir_path, 'vb.key'), 'r') as f:
        vestaboard_key = f.read().strip()
    return vestaboard_url, vestaboard_key

def call_vestaboard_api(characters_grid):
    url, key = load_vestaboard_config()
    headers = {"X-Vestaboard-Local-Api-Key": key, "Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(characters_grid), timeout=5)
        response.raise_for_status()
    except Exception as e:
        print(f"\n{get_timestamp()} [ERROR] Vestaboard Update Failed: {e}", flush=True)

def log_visitor(name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, 'visitor_log.txt')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a') as f:
        f.write(f"{timestamp} - {name}\n")

# --- CLICKABLE CAMERA SELECTOR ---

selected_index_global = None

def on_mouse_click(event, x, y, flags, param):
    global selected_index_global
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_cam_pos = x // 320
        available_indices = param 
        if clicked_cam_pos < len(available_indices):
            selected_index_global = available_indices[clicked_cam_pos]
            print(f"{get_timestamp()} [SUCCESS] Camera {selected_index_global} selected.", flush=True)

def select_camera_visual():
    global selected_index_global
    caps = []
    found_indices = []
    
    # Mute stderr for the hardware probe
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)

    try:
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                caps.append((i, cap))
                found_indices.append(i)
            else:
                cap.release()
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
    
    if not caps:
        print(f"{get_timestamp()} [FATAL] No cameras found.", flush=True)
        return 0
    
    if len(caps) == 1:
        print(f"{get_timestamp()} [INFO] Single camera detected. Auto-selecting Index {caps[0][0]}.", flush=True)
        caps[0][1].release()
        return caps[0][0]

    cv2.namedWindow("Click to Select Camera")
    cv2.setMouseCallback("Click to Select Camera", on_mouse_click, param=found_indices)
    print(f"{get_timestamp()} [INFO] Awaiting camera selection in popup window...", flush=True)
    
    while selected_index_global is None:
        previews = []
        for index, cap in caps:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (320, 240))
                cv2.putText(frame, f"SELECT CAM {index}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                previews.append(frame)
        
        if previews:
            combined = np.hstack(previews)
            cv2.imshow("Click to Select Camera", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            selected_index_global = found_indices[0]
            break

    for index, cap in caps:
        cap.release()
    cv2.destroyWindow("Click to Select Camera")
    
    choice = selected_index_global
    selected_index_global = None 
    return choice

# --- MAIN LOGIC ---

session_visitors = set()

if os.path.exists("known_faces.pkl"):
    with open("known_faces.pkl", "rb") as f:
        known_faces = pickle.load(f)
else:
    known_faces = {"encodings": [], "names": []}

selected_cam = select_camera_visual()
video_capture = cv2.VideoCapture(selected_cam)
last_seen = {}
is_prompting = False 

print(f"{get_timestamp()} [SYSTEM] VestaFace initializing on Camera {selected_cam}...", flush=True)
print(f"{get_timestamp()} [INFO] Press Ctrl+C to stop.", flush=True)

try:
    while True:
        ret, frame = video_capture.read()
        if not ret: break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_input = np.ascontiguousarray(rgb_small_frame)
        
        face_locations = face_recognition.face_locations(face_input)
        face_encodings = face_recognition.face_encodings(face_input, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            if known_faces["encodings"]:
                matches = face_recognition.compare_faces(known_faces["encodings"], face_encoding, tolerance=0.5)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_faces["names"][first_match_index]

            if name == "Unknown" and not is_prompting:
                is_prompting = True
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                new_name = simpledialog.askstring("Enrollment", "Enter name:")
                root.destroy()
                
                if new_name and new_name.strip():
                    known_faces["encodings"].append(face_encoding)
                    known_faces["names"].append(new_name.strip())
                    with open("known_faces.pkl", "wb") as f:
                        pickle.dump(known_faces, f)
                    name = new_name.strip()
                    print(f"{get_timestamp()} [NEW USER] {name} enrolled.", flush=True)
                is_prompting = False

            if name != "Unknown":
                session_visitors.add(name)
                current_time = time.time()
                
                # Terminal log
                print(f"{get_timestamp()} [FACE SPOT] Recognized: {name}", flush=True)

                if name not in last_seen or (current_time - last_seen[name] > 300):
                    log_visitor(name)
                    
                    grid = [[0 for _ in range(22)] for _ in range(6)]
                    
                    # Convert text to Vestaboard grid
                    def get_centered_string(text):
                        text = text.upper().strip()
                        padding = (22 - len(text)) // 2
                        return (" " * padding) + text

                    def text_to_codes(text):
                        # Vestaboard Char Map
                        char_map = {" ": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20, "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26, "1": 27, "2": 28, "3": 29, "4": 30, "5": 31, "6": 32, "7": 33, "8": 34, "9": 35, "0": 36, "!": 37, "@": 38, "#": 39, "$": 40, "(": 41, ")": 42, "-": 44, "+": 46, "&": 47, "=": 48, ";": 49, ":": 50, "'": 52, '"': 53, "%": 54, ",": 55, ".": 56, "/": 59, "?": 60}
                        text = text.upper().ljust(22)[:22]
                        return [char_map.get(char, 0) for char in text]

                    grid[1] = text_to_codes(get_centered_string(f"HELLO {name}"))
                    grid[3] = text_to_codes(get_centered_string("WELCOME TO THE"))
                    grid[4] = text_to_codes(get_centered_string("VESTASCRIPTERS"))
                    grid[5] = text_to_codes(get_centered_string("HEADQUARTERS"))
                    
                    print(f"{get_timestamp()} [UPDATE] Updating Vestaboard for {name}.", flush=True)
                    threading.Thread(target=call_vestaboard_api, args=(grid,), daemon=True).start()
                    last_seen[name] = current_time

        cv2.imshow('VestaFace Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print(f"\n{get_timestamp()} [STOP] Shutdown signal received.", flush=True)

finally:
    # Final Report
    summary = f"Session Total: {len(session_visitors)} people seen ({', '.join(session_visitors) if session_visitors else 'None'})"
    print(f"{get_timestamp()} [REPORT] {summary}", flush=True)
    
    # Release camera and close windows
    if video_capture.isOpened():
        video_capture.release()
    cv2.destroyAllWindows()
    print(f"{get_timestamp()} [CLOSED] Resources released.", flush=True)