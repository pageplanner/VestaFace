import os
import sys
import warnings
import winsound 

# Silence warnings and backends
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["VIDEOIO_PRIORITY_MSMF"] = "0" 

import face_recognition
import cv2
import pickle
import tkinter as tk
from tkinter import simpledialog, messagebox
import numpy as np
import time
import json
import requests
import threading
from datetime import datetime

# --- CONFIGURATION ---
VESTA_POST_UPDATE_DELAY = 30
GREETING_COOLDOWN = 300     
ENROLL_TIMEOUT = 60        
DATA_FILE = "known_faces.pkl"
SETTINGS_FILE = "settings.json"
PHOTO_DIR = "visitor_photos"

if not os.path.exists(PHOTO_DIR):
    os.makedirs(PHOTO_DIR)

# --- TRACKING ---
session_stats = {} 

# --- HELPERS ---

def get_timestamp():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def print_log(msg):
    print(f"{get_timestamp()} {msg}", flush=True)

def log_visitor(name, event_type="VISIT"):
    if name not in session_stats:
        session_stats[name] = {'count': 0, 'enrolled': False}
    session_stats[name]['count'] += 1
    if event_type == "ENROLLMENT":
        session_stats[name]['enrolled'] = True

    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, 'visitor_log.txt')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_path, 'a') as f:
            f.write(f"{timestamp} [{event_type}] - {name}\n")
    except Exception as e:
        print_log(f"[ERROR] Logging failed: {e}")

def save_screenshot(frame, name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.jpg"
    filepath = os.path.join(PHOTO_DIR, filename)
    cv2.imwrite(filepath, frame)
    print_log(f"[PHOTO] Saved screenshot: {filename}")

def save_settings(zoom, tolerance):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump({"zoom": zoom, "tolerance": tolerance}, f)
    except: pass 

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
                return data.get("zoom", 15), data.get("tolerance", 50)
        except: pass
    return 15, 50

# --- VESTABOARD LOGIC ---

def load_vestaboard_config():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'vb.url'), 'r') as f:
        url = f.read().strip()
    with open(os.path.join(dir_path, 'vb.key'), 'r') as f:
        key = f.read().strip()
    return url, key

def get_grid_row(text):
    text = text.upper().strip()
    pad = (22 - len(text)) // 2
    txt = ((" " * pad) + text).ljust(22)[:22]
    cmap = {" ": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20, "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26, "1": 27, "2": 28, "3": 29, "4": 30, "5": 31, "6": 32, "7": 33, "8": 34, "9": 35, "0": 36, "!": 37, "@": 38, "#": 39, "$": 40, "(": 41, ")": 42, "-": 44, "+": 46, "&": 47, "=": 48, ";": 49, ":": 50, "'": 52, '"': 53, "%": 54, ",": 55, ".": 56, "/": 59, "?": 60}
    return [cmap.get(c, 0) for c in txt]

def call_vestaboard_api(grid):
    url, key = load_vestaboard_config()
    headers = {"X-Vestaboard-Local-Api-Key": key, "Content-Type": "application/json"}
    try:
        requests.post(url, headers=headers, data=json.dumps(grid), timeout=5).raise_for_status()
    except Exception as e:
        print_log(f"[ERROR] Vestaboard API: {e}")

def clear_to_standby():
    grid = [[0 for _ in range(22)] for _ in range(6)]
    grid[0], grid[1], grid[2] = get_grid_row("VESTAFACE"), get_grid_row("BY"), get_grid_row("VESTASCRIPTERS")
    grid[4], grid[5] = get_grid_row("LOVEMYBOARD@"), get_grid_row("VESTASCRIPTERS.COM")
    print_log("[SYSTEM] Resetting board to standby layout.")
    threading.Thread(target=call_vestaboard_api, args=(grid,), daemon=True).start()

def push_greeting(name):
    grid = [[0 for _ in range(22)] for _ in range(6)]
    grid[1], grid[3], grid[4], grid[5] = get_grid_row(f"HELLO {name}"), get_grid_row("WELCOME TO THE"), get_grid_row("VESTASCRIPTERS"), get_grid_row("HEADQUARTERS")
    print_log(f"[UPDATE] Greeting {name}.")
    threading.Thread(target=call_vestaboard_api, args=(grid,), daemon=True).start()

# --- CAMERA SELECTOR ---

selected_index_global = None

def on_mouse_click(event, x, y, flags, param):
    global selected_index_global
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_cam_pos = x // 320
        if clicked_cam_pos < len(param):
            selected_index_global = param[clicked_cam_pos]

def select_camera_visual():
    global selected_index_global
    caps, found = [], []
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened(): caps.append((i, cap)); found.append(i)
        else: cap.release()
    if len(found) <= 1: return found[0] if found else 0
    cv2.namedWindow("Select Camera", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Camera", on_mouse_click, param=found)
    print_log("[INFO] Awaiting camera selection...")
    while selected_index_global is None:
        previews = []
        for index, cap in caps:
            ret, frame = cap.read()
            if ret:
                f = cv2.resize(frame, (320, 240))
                cv2.putText(f, f"CAM {index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                previews.append(f)
        if previews: cv2.imshow("Select Camera", np.hstack(previews))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    for _, cap in caps: cap.release()
    cv2.destroyWindow("Select Camera")
    return selected_index_global if selected_index_global is not None else 0

# --- MAIN LOOP ---

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f: known_faces = pickle.load(f)
else: known_faces = {"encodings": [], "names": []}

selected_cam = select_camera_visual()
video_capture = cv2.VideoCapture(selected_cam)
last_seen, last_vesta_update_time = {}, 0
last_unknown_prompt_time = 0
is_prompting = False
standby_pushed = True

saved_zoom, saved_tol = load_settings()
cv2.namedWindow('VestaFace Monitor', cv2.WINDOW_NORMAL)

def on_slider_change(x):
    try:
        z = cv2.getTrackbarPos('Zoom', 'VestaFace Monitor')
        t = cv2.getTrackbarPos('Tolerance', 'VestaFace Monitor')
        if z != -1 and t != -1: save_settings(z, t)
    except: pass

cv2.createTrackbar('Zoom', 'VestaFace Monitor', saved_zoom, 40, on_slider_change)
cv2.createTrackbar('Tolerance', 'VestaFace Monitor', saved_tol, 100, on_slider_change)

print_log(f"[SYSTEM] VestaFace active on Camera {selected_cam}...")

try:
    while True:
        ret, frame = video_capture.read()
        if not ret: break

        zoom_val = cv2.getTrackbarPos('Zoom', 'VestaFace Monitor')
        current_zoom = max(1.0, zoom_val / 10.0) if zoom_val > 0 else 1.0
        tol_val = cv2.getTrackbarPos('Tolerance', 'VestaFace Monitor')
        current_tolerance = (tol_val / 100.0) if tol_val > 0 else 0.5

        if current_zoom > 1.0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w / current_zoom), int(h / current_zoom)
            x1, y1 = (w - new_w) // 2, (h - new_h) // 2
            frame = cv2.resize(frame[y1:y1+new_h, x1:x1+new_w], (w, h))

        current_time = time.time()
        time_since_update = current_time - last_vesta_update_time
        
        rgb_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)[:, :, ::-1]
        rgb_small = np.ascontiguousarray(rgb_small) 
        face_locs = face_recognition.face_locations(rgb_small)

        if is_prompting:
            # UI OVERLAY: WAITING STATUS - Updated to Blue (BGR: 255, 0, 0)
            cv2.putText(frame, "WAITING FOR INPUT", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.circle(frame, (30, 30), 10, (255, 0, 0), -1)
        elif time_since_update > VESTA_POST_UPDATE_DELAY:
            if not standby_pushed:
                clear_to_standby()
                standby_pushed = True

            cv2.circle(frame, (30, 30), 10, (0, 255, 0), -1)
            cv2.putText(frame, "SEARCHING", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            face_encs = face_recognition.face_encodings(rgb_small, face_locs)

            for (top, right, bottom, left), face_encoding in zip(face_locs, face_encs):
                cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 255, 0), 2)
                
                name = "Unknown"
                if known_faces["encodings"]:
                    matches = face_recognition.compare_faces(known_faces["encodings"], face_encoding, tolerance=current_tolerance)
                    if True in matches: name = known_faces["names"][matches.index(True)]

                if name == "Unknown" and not is_prompting:
                    if current_time - last_unknown_prompt_time > GREETING_COOLDOWN:
                        is_prompting = True
                        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                        
                        def get_user_input(container):
                            root = tk.Tk()
                            root.withdraw()
                            root.attributes("-topmost", True)
                            def safe_destroy():
                                try:
                                    if root.winfo_exists(): root.destroy()
                                except: pass
                            root.after(ENROLL_TIMEOUT * 1000, safe_destroy)
                            container['name'] = simpledialog.askstring("Enrollment", "Enter name:", parent=root)
                            safe_destroy()

                        input_container = {'name': None}
                        thread = threading.Thread(target=get_user_input, args=(input_container,))
                        thread.start()
                        
                        start_wait = time.time()
                        while thread.is_alive():
                            ret_wait, frame_wait = video_capture.read()
                            if ret_wait:
                                if current_zoom > 1.0:
                                    frame_wait = cv2.resize(frame_wait[y1:y1+new_h, x1:x1+new_w], (w, h))
                                # Temporary frame while waiting - Updated to Blue
                                cv2.putText(frame_wait, "WAITING FOR INPUT", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                                cv2.circle(frame_wait, (30, 30), 10, (255, 0, 0), -1)
                                cv2.imshow('VestaFace Monitor', frame_wait)
                                cv2.waitKey(1)
                            if time.time() - start_wait > ENROLL_TIMEOUT + 2: break

                        new_name = input_container['name']
                        if new_name and new_name.strip():
                            name = new_name.strip()
                            known_faces["encodings"].append(face_encoding)
                            known_faces["names"].append(name)
                            with open(DATA_FILE, "wb") as f: pickle.dump(known_faces, f)
                            log_visitor(name, "ENROLLMENT")
                            save_screenshot(frame, name)
                            push_greeting(name)
                            last_seen[name], last_vesta_update_time, standby_pushed = time.time(), time.time(), False
                        else:
                            print_log("[TIMEOUT/CANCEL] No name entered.")
                            last_unknown_prompt_time = time.time()
                        
                        is_prompting = False
                        break 

                if name != "Unknown":
                    if name not in last_seen or (time.time() - last_seen[name] > GREETING_COOLDOWN):
                        log_visitor(name)
                        save_screenshot(frame, name)
                        push_greeting(name)
                        last_seen[name], last_vesta_update_time, standby_pushed = time.time(), time.time(), False
                        break 
        else:
            rem = int(VESTA_POST_UPDATE_DELAY - time_since_update)
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"Face Seen ({rem}s)", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for (top, right, bottom, left) in face_locs:
                cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 1)

        cv2.putText(frame, f"Z: {current_zoom:.1f}x  T: {current_tolerance:.2f}", (frame.shape[1]-180, frame.shape[0]-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('VestaFace Monitor', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('s'): save_screenshot(frame, "MANUAL")

except KeyboardInterrupt: pass
finally:
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "="*50)
    print(" VESTAFACE SESSION SUMMARY ".center(50, "="))
    print("="*50)
    print(f"{'NAME':<25} | {'VISITS':<8} | {'NEW?'}")
    print("-" * 50)
    for name, stats in sorted(session_stats.items()):
        new_tag = "YES" if stats['enrolled'] else "No"
        print(f"{name:<25} | {stats['count']:<8} | {new_tag}")
    print("="*50)
    print(f"Total Visitors Handled: {len(session_stats)}")
    print("="*50 + "\n")

    if video_capture.isOpened(): video_capture.release()
    cv2.destroyAllWindows()