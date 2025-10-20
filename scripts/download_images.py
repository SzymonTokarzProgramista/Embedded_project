#!/usr/bin/env python3
# download_monster_dataset.py
# Pobiera zdjęcia puszek Monster Energy, filtruje i usuwa duplikaty.
# Uwaga: sprawdź licencje pobranych obrazów przed użyciem w projekcie komercyjnym.

import os
import time
from pathlib import Path
from icrawler.builtin import BingImageCrawler
from PIL import Image, UnidentifiedImageError
import imagehash
from tqdm import tqdm

# ---------- USTAWIENIA ----------
OUT_DIR = Path("dataset/monster_energy")
TMP_DIR = OUT_DIR / "raw"
FINAL_DIR = OUT_DIR / "final"
QUERIES = [
    "Monster Energy can",
    "Monster energy drink can",
    "Monster Energy can close up",
    "Monster energy can hand",
    "Monster energy can on table",
    "Monster Energy logo can",
    "Monster energy aluminum can",
    "Monster energy drink can opened",
    "Monster energy can pack",
    "Monster energy can outdoors"
]
# ile obrazów w sumie chcemy mieć (po deduplikacji możesz nie osiągnąć dokładnie tej liczby)
TARGET_TOTAL = 5000
# ile maksymalnie pobrać na jedno zapytanie (można zwiększyć)
PER_QUERY_MAX = 700
# minimalny akceptowalny rozmiar (px) - obrazy mniejsze będą odrzucane
MIN_WIDTH = 100
MIN_HEIGHT = 100
# czy przeskalować finalne obrazy (np. do 224x224)
RESIZE_FINAL = True
FINAL_SIZE = (224, 224)
# przerwa między żądaniami (dla mniej agresywnego pobierania)
SLEEP_BETWEEN_QUERIES = 2.0
# ---------------------------------

TMP_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

def crawl_query(query, save_dir, max_num):
    print(f"\n[*] Crawling '{query}' -> {max_num} images")
    crawler = BingImageCrawler(
        storage={'root_dir': str(save_dir)},
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4
    )
    try:
        crawler.crawl(keyword=query, max_num=max_num)
    except Exception as e:
        print("Błąd podczas crawl:", e)

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()  # sprawdza integralność
        with Image.open(path) as img:
            w, h = img.size
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                return False
        return True
    except (UnidentifiedImageError, OSError, IOError):
        return False

def dedupe_and_save(src_dir, dest_dir, resize=False, final_size=(224,224)):
    print("\n[*] Usuwanie duplikatów i kopiowanie do finalnego katalogu...")
    hashes = {}
    files = sorted([p for p in Path(src_dir).rglob("*") if p.is_file()])
    kept = 0
    for p in tqdm(files):
        if not is_valid_image(p):
            # usuń uszkodzone / za małe
            try:
                p.unlink()
            except Exception:
                pass
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        h = imagehash.phash(img)
        # jeśli hash już istnieje, pomiń (duplikat)
        if str(h) in hashes:
            continue
        hashes[str(h)] = p
        # zapisz/lub skopiuj do final
        dest_path = Path(dest_dir) / f"{kept:06d}.jpg"
        if resize:
            img = img.resize(final_size, Image.LANCZOS)
        try:
            img.save(dest_path, "JPEG", quality=90)
            kept += 1
        except Exception as e:
            print("Błąd zapisu:", e)
        # opcjonalnie zatrzymaj gdy mamy wystarczająco
    print(f"Zachowanych obrazów: {kept}")
    return kept

def main():
    # policz ile pobrać z każdego zapytania by zbliżyć się do TARGET_TOTAL
    per_query = min(PER_QUERY_MAX, max(50, TARGET_TOTAL // max(1, len(QUERIES))))
    print(f"Plan pobierania: ~{per_query} obrazów na zapytanie ({len(QUERIES)} zapytań).")

    # crawl
    for q in QUERIES:
        q_safe = q.replace("/", "_").replace(" ", "_")
        dir_for_q = TMP_DIR / q_safe
        dir_for_q.mkdir(parents=True, exist_ok=True)
        crawl_query(q, dir_for_q, per_query)
        time.sleep(SLEEP_BETWEEN_QUERIES)

    # deduplikacja zebranych plików (scal wszystko z podfolderów tmp do final)
    # najpierw przenieś wszystkie pliki z podfolderów do jednego folderu temp_flat
    flat = TMP_DIR / "flat_all"
    flat.mkdir(parents=True, exist_ok=True)
    for p in TMP_DIR.rglob("*"):
        if p.is_file():
            try:
                # unikaj nadpisania - nadaj unikalną nazwę
                dest = flat / (p.stem + p.suffix)
                i = 1
                while dest.exists():
                    dest = flat / f"{p.stem}_{i}{p.suffix}"
                    i += 1
                p.rename(dest)
            except Exception:
                pass

    kept = dedupe_and_save(flat, FINAL_DIR, resize=RESIZE_FINAL, final_size=FINAL_SIZE)
    print("\nGotowe.")
    print(f"Finalna liczba obrazów w {FINAL_DIR}: {kept}")
    print("Jeśli potrzebujesz dokładnie 5000 obrazów, spróbuj zwiększyć PER_QUERY_MAX lub dodać więcej zapytań.")

if __name__ == "__main__":
    main()
