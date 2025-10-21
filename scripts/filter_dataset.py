import os
import glob
from pathlib import Path

# 🔧 USTAWIENIA
dataset_dir = "dataset/Embedded_Systems.v1i.yolov8"  # katalog z datasetem YOLO
splits = ["train", "valid", "test"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

removed = 0
skipped = 0

def find_image(images_dir: str, stem: str):
    """Znajdź obraz pasujący do nazwy pliku etykiety (bez rozszerzenia)."""
    for ext in IMAGE_EXTS + [e.upper() for e in IMAGE_EXTS]:
        path = os.path.join(images_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


for split in splits:
    labels_dir = os.path.join(dataset_dir, split, "labels")
    images_dir = os.path.join(dataset_dir, split, "images")
    if not os.path.isdir(labels_dir):
        print(f"[{split}] Brak katalogu labels – pomijam.")
        continue

    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    print(f"[{split}] Przetwarzam {len(label_files)} plików etykiet...")

    for lbl_path in label_files:
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:  # brak jakichkolwiek etykiet (pusty plik)
            base = os.path.splitext(os.path.basename(lbl_path))[0]
            img_path = find_image(images_dir, base)
            if img_path and os.path.exists(img_path):
                os.remove(img_path)
            os.remove(lbl_path)
            removed += 1
        else:
            skipped += 1

print("============================================")
print(f"Usunięto {removed} pustych przykładów (obrazy + etykiety).")
print(f"Pozostawiono {skipped} plików etykiet z danymi.")
print("============================================")
