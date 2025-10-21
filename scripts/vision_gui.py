"""
Tkinter YOLO Quick Test – real-time, bez wątków (używa .after()).
- Wczytaj swój .pt (Ultralytics YOLO)
- Start/Stop kamery z wyborem indeksu
- Live predict na każdej klatce (checkbox)
- Suwak conf
"""

import sys, time, platform
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Quick Test – Tkinter (real-time)")

        # YOLO
        self.model = None
        self.model_path = None
        self.class_names = {}
        self.conf = tk.DoubleVar(value=0.25)
        self.live = tk.BooleanVar(value=True)

        # Kamera
        self.cap = None
        self.cam_running = False
        self.cam_index = tk.IntVar(value=0)

        # FPS
        self._last_t = None
        self._fps = 0.0

        # UI top (kamera)
        top = tk.Frame(root)
        top.pack(fill=tk.X, padx=8, pady=(8, 0))
        tk.Label(top, text="Kamera:").pack(side=tk.LEFT)
        tk.Spinbox(top, from_=0, to=10, width=4, textvariable=self.cam_index).pack(side=tk.LEFT, padx=6)

        # Podgląd
        self.panel = tk.Label(root, text="Podgląd", bg="#222", fg="#ddd")
        self.panel.pack(padx=8, pady=8, fill=tk.BOTH, expand=True)
        self._tk_img = None  # referencja do PhotoImage

        # Przyciski
        btns = tk.Frame(root)
        btns.pack(fill=tk.X, padx=8, pady=(0, 6))

        tk.Button(btns, text="Wczytaj .pt", command=self.load_model).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Start kamera", command=self.start_camera).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Stop kamera", command=self.stop_camera).pack(side=tk.LEFT, padx=4)
        tk.Label(btns, text="conf:").pack(side=tk.LEFT, padx=(16, 2))
        tk.Scale(btns, from_=0.05, to=0.95, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=self.conf, length=180).pack(side=tk.LEFT)
        tk.Checkbutton(btns, text="Live predict", variable=self.live).pack(side=tk.LEFT, padx=(16, 4))

        tk.Button(btns, text="Otwórz obraz", command=self.open_image).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Predict (1x)", command=self.predict_once).pack(side=tk.LEFT, padx=4)

        # Status
        self.status = tk.Label(root, text="Model: (brak) | FPS: --", anchor="w")
        self.status.pack(fill=tk.X, padx=8, pady=(0, 8))

        # Czcionka PIL
        try:
            self.font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            self.font = ImageFont.load_default()

        # Pętla odświeżania (co ~15 ms)
        self.root.after(15, self._update_loop)

        # Zamknięcie
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # === YOLO ===
    def load_model(self):
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
            self.model = None
            self.model_path = None

    # === Kamera ===
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
        self._last_t = time.time()

    def stop_camera(self):
        self.cam_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.status.config(text="Kamera zatrzymana")

    # === Obraz z pliku ===
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
        self.show_bgr(img_bgr)

    # === Pętla odświeżania GUI ===
    def _update_loop(self):
        if self.cam_running and self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                out = frame
                if self.live.get() and self.model is not None:
                    try:
                        out = self._infer_and_draw(frame)
                    except Exception as e:
                        # pokaż klatkę bez detekcji + błąd w statusie
                        out = frame
                        self.status.config(
                            text=f"Model: {Path(self.model_path).name if self.model_path else '(brak)'} | Błąd: {e}"
                        )

                # FPS
                now = time.time()
                dt = (now - self._last_t) if self._last_t else 0.0
                self._last_t = now
                if dt > 0:
                    self._fps = 1.0 / dt

                self.show_bgr(out)
                self.status.config(
                    text=f"Model: {Path(self.model_path).name if self.model_path else '(brak)'} | FPS: {self._fps:0.1f}"
                )
            else:
                self.status.config(text="Brak klatki z kamery…")

        # planuj następny tick
        self.root.after(15, self._update_loop)

    # === Jednorazowa predykcja na aktualnym obrazie (jeśli jest) ===
    def predict_once(self):
        if self.model is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj model .pt")
            return
        if self.cap is None:
            messagebox.showwarning("Uwaga", "Kamera nie działa — uruchom ją albo wczytaj obraz")
            return
        ok, frame = self.cap.read()
        if not ok:
            messagebox.showwarning("Uwaga", "Brak klatki z kamery")
            return
        out = self._infer_and_draw(frame)
        self.show_bgr(out)

    # === Inference + rysowanie ===
    def _infer_and_draw(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=rgb, conf=float(self.conf.get()), iou = 0.4, verbose=False)
        r = results[0]

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            # Szybkie wyciąganie z tensorów
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
                tw = draw.textlength(txt, font=self.font)
                th = getattr(self.font, "size", 14) + 6
                y0 = max(0, y1 - th)
                draw.rectangle([x1, y0, x1 + int(tw) + 6, y1], fill=(0, 255, 0))
                draw.text((x1 + 3, y0 + 3), txt, fill=(0, 0, 0), font=self.font)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # === Wyświetlanie w Label ===
    def show_bgr(self, bgr):
        # dopasuj do aktualnych wymiarów panelu
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
        self.panel.config(image=self._tk_img, text="")  # czyści tekst "Podgląd"

    def on_close(self):
        self.stop_camera()
        self.panel.config(image="", text="Zamykam…")
        self._tk_img = None
        self.root.update_idletasks()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1000x700")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
