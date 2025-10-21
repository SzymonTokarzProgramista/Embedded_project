"""
Tkinter YOLO – responsywny podgląd + stabilny layout (grid).
- Toolbar (wiersz 0) – zawsze widoczny
- Podgląd (wiersz 1) – rozciąga się i skaluje obraz do ramki (nie zasłania przycisków)
- Status (wiersz 2)
- Kamera + inference w wątkach (GUI nie blokuje się)
- Domyślny model: runs/train/final_best2/weights/best.pt (auto-load jeśli istnieje)
- FPS rysowane na obrazie
"""

import time
import platform
import threading
from pathlib import Path
from queue import Queue, Empty

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

DEFAULT_MODEL_PATH = Path("runs/train/final_best2/weights/best.pt")


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Quick Test – Threaded + Stable Layout")
        self.root.minsize(950, 600)

        # === Ustawienia YOLO ===
        self.model = None
        self.model_path = None
        self.class_names = {}
        self.conf = tk.DoubleVar(value=0.25)
        self.live = tk.BooleanVar(value=True)

        # === Kamera / wątki ===
        self.cap = None
        self.cam_running = False
        self.cam_index = tk.IntVar(value=0)
        self.capture_thread = None
        self.infer_thread = None
        self.stop_event = threading.Event()

        # Bufory
        self.frame_queue = Queue(maxsize=1)   # drop stare, trzymaj najnowszą
        self.display_frame = None             # ostatnia ramka do wyświetlenia

        # FPS (liczone w wątku inferencyjnym)
        self._fps = 0.0
        self._t_last = None

        # Czcionka PIL do etykiet
        try:
            self.font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            self.font = ImageFont.load_default()

        # === Layout: 3 wiersze ===
        self.root.grid_rowconfigure(0, weight=0)  # toolbar
        self.root.grid_rowconfigure(1, weight=1)  # preview
        self.root.grid_rowconfigure(2, weight=0)  # status
        self.root.grid_columnconfigure(0, weight=1)

        # Toolbar (row=0)
        toolbar = tk.Frame(self.root)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        toolbar.grid_columnconfigure(99, weight=1)

        tk.Label(toolbar, text="Kamera:").grid(row=0, column=0, sticky="w")
        tk.Spinbox(toolbar, from_=0, to=10, width=4, textvariable=self.cam_index)\
            .grid(row=0, column=1, padx=(6, 12), sticky="w")

        tk.Button(toolbar, text="Wczytaj .pt", command=self.load_model_dialog)\
            .grid(row=0, column=2, padx=4, sticky="w")
        tk.Button(toolbar, text="Start kamera", command=self.start_camera)\
            .grid(row=0, column=3, padx=4, sticky="w")
        tk.Button(toolbar, text="Stop kamera", command=self.stop_camera)\
            .grid(row=0, column=4, padx=4, sticky="w")

        tk.Label(toolbar, text="conf:").grid(row=0, column=5, padx=(16, 2), sticky="e")
        tk.Scale(toolbar, from_=0.05, to=0.95, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=self.conf, length=180)\
            .grid(row=0, column=6, sticky="w")

        tk.Checkbutton(toolbar, text="Live predict", variable=self.live)\
            .grid(row=0, column=7, padx=(16, 4), sticky="w")

        tk.Button(toolbar, text="Otwórz obraz", command=self.open_image)\
            .grid(row=0, column=8, padx=4, sticky="w")
        tk.Button(toolbar, text="Predict (1x)", command=self.predict_once)\
            .grid(row=0, column=9, padx=4, sticky="w")

        # Preview (row=1)
        preview_frame = tk.Frame(self.root, bd=1, relief="sunken")
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        self.panel = tk.Label(preview_frame, text="Podgląd", bg="#222", fg="#ddd")
        self.panel.grid(row=0, column=0, sticky="nsew")
        self._tk_img = None

        # Status (row=2)
        self.status = tk.Label(self.root, text="Model: (brak) | FPS: --", anchor="w")
        self.status.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 8))

        # Autoload modelu
        self.try_load_default_model()

        # Pętla renderująca (tylko UI)
        self.root.after(15, self._ui_loop)

        # Skróty
        self.root.bind("<space>", lambda e: self.live.set(not self.live.get()))
        self.root.bind("<Escape>", lambda e: self.stop_camera())

        # Zamknięcie
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ===== Model =====
    def try_load_default_model(self):
        if YOLO is None:
            self.status.config(text="Brak ultralytics (pip install ultralytics)")
            return
        if DEFAULT_MODEL_PATH.exists():
            try:
                self.model = YOLO(str(DEFAULT_MODEL_PATH))
                self.model_path = str(DEFAULT_MODEL_PATH)
                self.class_names = getattr(self.model, "names", {}) or {}
                self.status.config(text=f"Model: {DEFAULT_MODEL_PATH.name} | FPS: -- (auto)")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się załadować domyślnego modelu:\n{e}")
        else:
            self.status.config(text=f"Brak modelu: {DEFAULT_MODEL_PATH}")

    def load_model_dialog(self):
        if YOLO is None:
            messagebox.showerror("Błąd", "Brak ultralytics. Zainstaluj: pip install ultralytics")
            return
        path = filedialog.askopenfilename(
            title="Wybierz plik modelu .pt",
            filetypes=[("YOLO model", "*.pt"), ("Wszystkie pliki", "*.*")]
        )
        if not path:
            return
        try:
            self.model = YOLO(path)
            self.model_path = path
            self.class_names = getattr(self.model, "names", {}) or {}
            self.status.config(text=f"Model: {Path(path).name} | FPS: --")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się załadować modelu:\n{e}")

    # ===== Kamera =====
    def start_camera(self):
        if self.cam_running:
            return
        idx = int(self.cam_index.get())
        if platform.system().lower().startswith("win"):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            messagebox.showerror("Błąd", f"Nie mogę otworzyć kamery {idx}")
            return

        self.cap = cap
        self.cam_running = True
        self.stop_event.clear()

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        self.infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self.infer_thread.start()

        self.status.config(text=f"Kamera {idx} START")

    def stop_camera(self):
        self.cam_running = False
        self.stop_event.set()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.status.config(text="Kamera zatrzymana")

    # ===== Wątki =====
    def _capture_loop(self):
        while not self.stop_event.is_set():
            if self.cap is None:
                break
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            # drop stary wpis w kolejce, trzymaj tylko najnowszą klatkę
            if not self.frame_queue.empty():
                try:
                    _ = self.frame_queue.get_nowait()
                except Empty:
                    pass
            self.frame_queue.put(frame)
            time.sleep(0.001)

    def _infer_loop(self):
        self._t_last = time.time()
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except Empty:
                continue

            out = frame
            if self.live.get() and self.model is not None:
                try:
                    out = self._infer_and_draw(frame)
                except Exception as e:
                    out = frame
                    self.status.config(text=f"Błąd detekcji: {e}")

            # FPS
            now = time.time()
            dt = now - (self._t_last or now)
            self._t_last = now
            if dt > 0:
                self._fps = 1.0 / dt

            # FPS overlay
            cv2.putText(out, f"{self._fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

            self.display_frame = out

    # ===== Otwieranie obrazu z pliku =====
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[("Obrazy", "*.png;*.jpg;*.jpeg;*.bmp"), ("Wszystkie pliki", "*.*")]
        )
        if not path:
            return
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("Błąd", "Nie udało się wczytać obrazu")
            return
        # pokaż obraz bez kamery; po włączeniu kamery podgląd ją nadpisze
        self.display_frame = img_bgr
        self._show_bgr(img_bgr)

    # ===== UI render loop =====
    def _ui_loop(self):
        if self.display_frame is not None:
            self._show_bgr(self.display_frame)
            self.status.config(text=f"Model: {Path(self.model_path).name if self.model_path else '(brak)'}")
        self.root.after(15, self._ui_loop)

    # ===== Predykcja 1x =====
    def predict_once(self):
        if self.model is None:
            messagebox.showwarning("Uwaga", "Najpierw załaduj model .pt")
            return
        frame = None
        try:
            frame = self.frame_queue.get_nowait()
        except Empty:
            frame = self.display_frame
        if frame is None:
            messagebox.showwarning("Uwaga", "Brak klatki z kamery – uruchom kamerę albo wczytaj obraz")
            return
        try:
            out = self._infer_and_draw(frame.copy())
            cv2.putText(out, f"{self._fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            self.display_frame = out
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    # ===== Inference + rysowanie =====
    def _infer_and_draw(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=rgb, conf=float(self.conf.get()), verbose=False)
        r = results[0]

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy
            confs = boxes.conf
            clss = boxes.cls
            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu().numpy()
                confs = confs.cpu().numpy()
                clss = clss.cpu().numpy().astype(int)
            else:
                xyxy = np.array(xyxy)
                confs = np.array(confs)
                clss = np.array(clss).astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = self.class_names.get(int(k), str(int(k)))
                txt = f"{label} {float(c):.2f}"
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                try:
                    tw = draw.textlength(txt, font=self.font)
                except Exception:
                    tw = len(txt) * 7
                th = getattr(self.font, "size", 14) + 6
                y0 = max(0, y1 - th)
                draw.rectangle([x1, y0, x1 + int(tw) + 6, y1], fill=(0, 255, 0))
                draw.text((x1 + 3, y0 + 3), txt, fill=(0, 0, 0), font=self.font)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ===== Wyświetlanie =====
    def _show_bgr(self, bgr):
        # skaluj do rozmiaru panelu w ramce preview (nie powiększaj layoutu)
        self.panel.update_idletasks()
        max_w = max(100, self.panel.winfo_width())
        max_h = max(100, self.panel.winfo_height())
        h, w = bgr.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        self._tk_img = ImageTk.PhotoImage(im)
        self.panel.config(image=self._tk_img, text="")

    # ===== Zamknięcie =====
    def on_close(self):
        self.stop_camera()
        self.stop_event.set()
        self.panel.config(image="", text="Zamykam…")
        self._tk_img = None
        self.root.update_idletasks()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
