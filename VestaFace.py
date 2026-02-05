import cv2
import numpy as np
import os
import time
import threading
import requests
import json
import shutil
import tkinter as tk
from tkinter import ttk, messagebox
from insightface.app import FaceAnalysis
from datetime import datetime

# --- CONFIGURATION ---
RE_GREET_DELAY = 60         
BOARD_RESET_DELAY = 30      
SIMILARITY_THRESHOLD = 0.45 
ENROLL_TRIGGER_SIZE = 80    
PROMPT_TIMEOUT = 30         
DATA_FILE = "face_db.npz"
LOG_FILE = "visitor_log.txt"
BACKUP_DIR = "backups"

# --- GLOBAL TRACKING ---
session_stats = {} 
visitor_history = [] 

def get_timestamp():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def print_log(msg):
    print(f"{get_timestamp()} {msg}", flush=True)

class VestaFace:
    def __init__(self, name='buffalo_l', db_path=DATA_FILE):
        print_log("--- Step 2: Initializing ArcFace Model ---")
        self.app = FaceAnalysis(name=name, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, 'vb.url'), 'r') as f: self.vb_url = f.read().strip()
        with open(os.path.join(dir_path, 'vb.key'), 'r') as f: self.vb_key = f.read().strip()
        
        self.db_path = db_path
        self.known_embeddings, self.known_names = [], []
        self.last_seen = {} 
        self.is_registering = False
        self.typing_buffer = ""
        self.current_enrollment_encoding = None
        self.prompt_start_time = 0
        self.status_msg = "SYSTEM READY - SCANNING"
        
        if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)
        self.load_database()

    def log_visitor(self, name, event_type="VISIT"):
        timestamp = get_timestamp()
        entry = f"{timestamp} [{event_type}] - {name}"
        visitor_history.append(entry)
        if name not in session_stats:
            session_stats[name] = {'count': 0, 'enrolled': (event_type == "ENROLLMENT")}
        session_stats[name]['count'] += 1
        with open(LOG_FILE, 'a') as f: f.write(entry + "\n")

    def get_grid_row(self, text):
        text = text.upper().strip()
        pad = (22 - len(text)) // 2
        txt = ((" " * pad) + text).ljust(22)[:22]
        cmap = {" ": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10, "K": 11, "L": 12, "M": 13, "N": 14, "O": 15, "P": 16, "Q": 17, "R": 18, "S": 19, "T": 20, "U": 21, "V": 22, "W": 23, "X": 24, "Y": 25, "Z": 26, "1": 27, "2": 28, "3": 29, "4": 30, "5": 31, "6": 32, "7": 33, "8": 34, "9": 35, "0": 36, "!": 37, "@": 38, "#": 39, "$": 40, "(": 41, ")": 42, "-": 44, "+": 46, "&": 47, "=": 48, ";": 49, ":": 50, "'": 52, '"': 53, "%": 54, ",": 55, ".": 56, "/": 59, "?": 60}
        return [cmap.get(c, 0) for c in txt]

    def call_vestaboard_api(self, grid):
        headers = {"X-Vestaboard-Local-Api-Key": self.vb_key, "Content-Type": "application/json"}
        try:
            requests.post(self.vb_url, headers=headers, data=json.dumps(grid), timeout=5)
            print_log("Local Board Update Success")
        except Exception as e:
            print_log(f"[ERROR] Vestaboard API: {e}")

    def push_greeting(self, name):
        grid = [[0 for _ in range(22)] for _ in range(6)]
        grid[1], grid[3], grid[4], grid[5] = self.get_grid_row(f"HELLO {name}"), self.get_grid_row("WELCOME TO THE"), self.get_grid_row("VESTASCRIPTERS"), self.get_grid_row("HEADQUARTERS")
        print_log(f"[GREETING] Pushing welcome message for: {name}")
        self.status_msg = f"GREETING SENT: {name.upper()}"
        threading.Thread(target=self.call_vestaboard_api, args=(grid,), daemon=True).start()

    def clear_to_standby(self):
        grid = [[0 for _ in range(22)] for _ in range(6)]
        grid[0], grid[1], grid[2] = self.get_grid_row("VESTAFACE"), self.get_grid_row("BY"), self.get_grid_row("VESTASCRIPTERS")
        grid[4], grid[5] = self.get_grid_row("LOVEMYBOARD@"), self.get_grid_row("VESTASCRIPTERS.COM")
        print_log("[STANDBY] Resetting Vestaboard layout.")
        self.status_msg = "SYSTEM READY - SCANNING"
        threading.Thread(target=self.call_vestaboard_api, args=(grid,), daemon=True).start()

    def load_database(self):
        if os.path.exists(self.db_path):
            data = np.load(self.db_path, allow_pickle=True)
            self.known_embeddings, self.known_names = list(data['embeddings']), list(data['names'])
            print_log(f"Database: {len(self.known_names)} faces loaded.")

    def delete_user(self, tree):
        selected_item = tree.selection()
        if not selected_item:
            messagebox.showwarning("Delete Error", "Please select a user first.")
            return
        user_name = tree.item(selected_item)['values'][0]
        if messagebox.askyesno("Confirm Delete", f"Permanently delete '{user_name}'?"):
            indices = [i for i, x in enumerate(self.known_names) if x == user_name]
            for index in sorted(indices, reverse=True):
                self.known_names.pop(index); self.known_embeddings.pop(index)
            np.savez(self.db_path, embeddings=self.known_embeddings, names=self.known_names)
            tree.delete(selected_item)

    def show_management(self):
        root = tk.Tk()
        root.title("VestaFace Admin Dashboard")
        root.geometry("650x580")
        root.attributes("-topmost", True)
        nb = ttk.Notebook(root); nb.pack(expand=True, fill='both', padx=5, pady=5)
        f1 = ttk.Frame(nb); nb.add(f1, text='Database & Stats')
        tree = ttk.Treeview(f1, columns=('Name', 'Visits', 'New'), show='headings')
        tree.heading('Name', text='Name'); tree.heading('Visits', text='Visits'); tree.heading('New', text='New')
        all_unique_names = sorted(list(set(self.known_names)))
        for name in all_unique_names:
            session_data = session_stats.get(name, {'count': 0, 'enrolled': False})
            tree.insert('', 'end', values=(name, session_data['count'], "Yes" if session_data['enrolled'] else "No"))
        tree.pack(expand=True, fill='both', padx=5, pady=5)
        btn_f = ttk.Frame(f1); btn_f.pack(fill='x', padx=5, pady=5)
        ttk.Button(btn_f, text="DELETE SELECTED USER", command=lambda: self.delete_user(tree)).pack(side='left', padx=2)
        ttk.Button(btn_f, text="RUN MANUAL BACKUP", command=self.run_backup).pack(side='right', padx=2)
        f2 = ttk.Frame(nb); nb.add(f2, text='History Log')
        txt = tk.Text(f2); scroll = ttk.Scrollbar(f2, command=txt.yview); txt.configure(yscrollcommand=scroll.set)
        for line in reversed(visitor_history): txt.insert('end', line + "\n")
        txt.pack(side='left', expand=True, fill='both'); scroll.pack(side='right', fill='y')
        root.mainloop()

    def run_backup(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if os.path.exists(DATA_FILE): shutil.copy(DATA_FILE, f"{BACKUP_DIR}/face_db_{ts}.npz")
            if os.path.exists(LOG_FILE): shutil.copy(LOG_FILE, f"{BACKUP_DIR}/visitor_log_{ts}.txt")
            print_log(f"[BACKUP] System archived.")
            messagebox.showinfo("Backup", "Manual Archive Successful.")
        except Exception as e: print_log(f"[ERROR] Backup failed: {e}")

    def process_frame(self, frame, zoom, key, stop_flag):
        # Fast exit: if stop_flag is already true, don't even process
        if stop_flag: return None

        if zoom > 100:
            h, w = frame.shape[:2]
            s = zoom / 100
            nh, nw = int(h/s), int(w/s)
            sy, sx = (h-nh)//2, (w-nw)//2
            frame = cv2.resize(frame[sy:sy+nh, sx:sx+nw], (w, h))

        h, w = frame.shape[:2]
        footer_height = 60
        ui_frame = np.zeros((h + footer_height, w, 3), dtype=np.uint8)
        ui_frame[0:h, 0:w] = frame

        if self.is_registering and (time.time() - self.prompt_start_time > PROMPT_TIMEOUT):
            print_log("[TIMEOUT] Enrollment cancelled.")
            self.is_registering = False; self.typing_buffer = ""

        if self.is_registering:
            self.status_msg = f"NEW VISITOR: {self.typing_buffer}_"
            if key == 13: # ENTER
                if self.typing_buffer.strip():
                    name = self.typing_buffer.strip()
                    self.known_embeddings.append(self.current_enrollment_encoding)
                    self.known_names.append(name)
                    np.savez(self.db_path, embeddings=self.known_embeddings, names=self.known_names)
                    print_log(f"[ENROLLED] New face saved: {name}")
                    self.log_visitor(name, "ENROLLMENT"); self.push_greeting(name)
                    threading.Timer(BOARD_RESET_DELAY, self.clear_to_standby).start()
                self.is_registering = False
            elif key == 8: self.typing_buffer = self.typing_buffer[:-1]
            elif 32 <= key <= 126: self.typing_buffer += chr(key)

        # Facial analysis - only run if not exiting
        faces = self.app.get(frame)
        for face in faces:
            if stop_flag: break # Break out if exit requested during processing
            name = "Unknown"; is_recognized = False
            if self.known_embeddings:
                sims = np.dot(self.known_embeddings, face.normed_embedding)
                idx = np.argmax(sims)
                if sims[idx] > SIMILARITY_THRESHOLD:
                    name = self.known_names[idx]; is_recognized = True
                    if self.is_registering: self.is_registering = False; self.typing_buffer = ""
                    now = time.time()
                    if not self.is_registering: self.status_msg = f"RECOGNIZED: {name.upper()}"
                    if name not in self.last_seen or (now - self.last_seen[name] > RE_GREET_DELAY):
                        print_log(f"[RECOGNIZED] Seen: {name}")
                        self.log_visitor(name); self.push_greeting(name)
                        self.last_seen[name] = now
                        threading.Timer(BOARD_RESET_DELAY, self.clear_to_standby).start()

            if not is_recognized and not self.is_registering and (face.bbox[2]-face.bbox[0]) > ENROLL_TRIGGER_SIZE:
                self.is_registering, self.typing_buffer = True, ""
                self.prompt_start_time = time.time(); self.current_enrollment_encoding = face.normed_embedding

            b = face.bbox.astype(int); color = (0, 255, 0) if is_recognized else (0, 165, 255)
            cv2.rectangle(ui_frame, (b[0], b[1]), (b[2], b[3]), color, 2)
            cv2.putText(ui_frame, name, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(ui_frame, self.status_msg, (20, h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # DRAW TOP-LEFT BUTTONS
        cv2.rectangle(ui_frame, (10, 10), (90, 35), (60, 60, 60), -1)
        cv2.putText(ui_frame, "ADMIN", (25, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.rectangle(ui_frame, (10, 45), (90, 70), (0, 0, 150), -1)
        cv2.putText(ui_frame, "EXIT", (28, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return ui_frame

def select_camera_grid():
    caps = []
    print_log("--- Step 1: Scanning for Cameras ---")
    for i in range(2): 
        c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if c.isOpened():
            ret, _ = c.read()
            if ret: caps.append((i, c))
            else: c.release()
    if not caps: return 0
    if len(caps) == 1: return caps[0][0]
    selected = [None]
    def sel_click(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            idx = x // 320
            if idx < len(caps): selected[0] = caps[idx][0]
    cv2.namedWindow("Select Camera"); cv2.setMouseCallback("Select Camera", sel_click)
    while selected[0] is None:
        pre = [cv2.resize(c.read()[1], (320, 240)) for idx, c in caps]
        cv2.imshow("Select Camera", np.hstack(pre))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    for _, c in caps: c.release()
    cv2.destroyWindow("Select Camera"); return selected[0]

if __name__ == "__main__":
    cam_id = select_camera_grid()
    vf = VestaFace(); cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    
    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 60
    scaled_w, scaled_h = int(original_w * 0.7), int(original_h * 0.7)
    
    window_name = 'VestaFace ArcFace V2'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, scaled_w, scaled_h)
    cv2.createTrackbar('Zoom', window_name, 100, 300, lambda x: None)
    
    stop_signal = [False]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            win_rect = cv2.getWindowImageRect(window_name)
            win_w, win_h = win_rect[2], win_rect[3]
            frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 60
            rx = x * (frame_w / win_w)
            ry = y * (frame_h / win_h)
            
            if 10 <= rx <= 90:
                if 10 <= ry <= 35:
                    threading.Thread(target=vf.show_management, daemon=True).start()
                elif 45 <= ry <= 70:
                    print_log("[SYSTEM] EXIT RECEIVED - TERMINATING IMMEDIATELY.")
                    stop_signal[0] = True

    cv2.setMouseCallback(window_name, on_mouse)

    while not stop_signal[0]:
        ret, frame = cap.read()
        if not ret or stop_signal[0]: break
        
        # Pass stop_signal to the processing function to abort heavy tasks
        ui = vf.process_frame(frame, cv2.getTrackbarPos('Zoom', window_name), cv2.waitKey(1) & 0xFF, stop_signal[0])
        
        if ui is not None:
            cv2.imshow(window_name, ui)
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print_log("[SYSTEM] Shut down successfully.")