#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balansowanie YOLOv8 przez augmentację klas niedoreprezentowanych + augmentacja negatywów.
Wersja z profilem augmentacji dopasowanym do zdjęć narzędzi w warunkach indoor (warsztat / LED,
odblaski metalu, delikatny motion blur, kompresja JPEG, cienie dłoni / obiektów).

Zmiany w stosunku do poprzedniej wersji:
- Zmieniony pipeline w `build_augment_pipeline()` (bardziej realistyczne fotometryczne i kamerowe artefakty,
  rozsądne geometrii, mniejsza agresja CoarseDropout, dodane ISONoise/Downscale/RandomShadow/SunFlare).
- `min_visibility` zmniejszone do 0.15, aby nie gubić małych obiektów (np. drill_bit).

Autor oryginału: Szymon Tokarz (rozszerzenia i poprawki)
"""

import argparse
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Tuple, Optional

import albumentations as A
import cv2
import numpy as np
import yaml
from tqdm import tqdm

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------- I/O pomocnicze ----------------------------

def read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_class_names(d):
    names = d.get("names")
    if isinstance(names, dict):
        # posortuj po ID klucza: {"0": "...", "1": "..."} lub {0: "..."}
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, list):
        return names
    raise ValueError("Nie udało się odczytać 'names' z data.yaml")

def yolo_txt_to_boxes(lbl_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Wczytaj YOLO .txt -> lista (cls, x, y, w, h) w norm. [0,1]. Pusta lista = negatyw."""
    out: List[Tuple[int, float, float, float, float]] = []
    if not lbl_path.exists():
        return out
    with lbl_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:5])
                    out.append((cls, x, y, w, h))
                except Exception:
                    continue
    return out

def boxes_to_yolo_txt(boxes: List[Tuple[int, float, float, float, float]]) -> str:
    return "\n".join(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in boxes)

def find_image_for_label(labels_dir: Path, images_dir: Path, lbl_path: Path) -> Optional[Path]:
    """
    Spróbuj dopasować obraz do etykiety zachowując relatywną ścieżkę:
    labels/.../name.txt  -> images/.../name.(ext)
    """
    try:
        rel = lbl_path.relative_to(labels_dir)          # np. sub/xyz.txt
        stem = rel.stem
        img_parent = images_dir / rel.parent
        for ext in IMAGE_EXTS:
            p = img_parent / f"{stem}{ext}"
            if p.exists():
                return p
        for ext in [e.upper() for e in IMAGE_EXTS]:
            p = img_parent / f"{stem}{ext}"
            if p.exists():
                return p
    except Exception:
        pass
    candidates = list(images_dir.rglob("*"))
    for p in candidates:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem == lbl_path.stem:
            return p
    return None

def count_instances(labels_dir: Path):
    """Zlicz instancje w całym drzewie splitu; zwróć (counts per class, map: plik -> set(klas))."""
    counts = Counter()
    file_to_classes: dict[Path, set[int]] = {}
    for p in labels_dir.rglob("*.txt"):
        boxes = yolo_txt_to_boxes(p)
        cls_ids = [b[0] for b in boxes]
        counts.update(cls_ids)
        file_to_classes[p] = set(cls_ids)
    return counts, file_to_classes


# ---------------------------- Augmentacje ----------------------------

def build_augment_pipeline():
    """
    Pipeline dopasowany do indoor (narzędzia):
    - Umiarkowana geometria (Affine / ShiftScaleRotate, Perspective bardzo małe) — zachowanie boxów.
    - Fotometryczne (brightness/contrast/gamma, HSV, RGB shift).
    - Artefakty kamerowe (GaussNoise/ISONoise, Motion/Gaussian/Median blur, kompresja JPEG, downscale).
    - Cienie/odblaski (RandomShadow, SunFlare), CLAHE na kontrast w cieniach.
    - CoarseDropout delikatniejsze, by nie zasłaniać całych małych obiektów (np. drill_bit).
    """
    return A.Compose(
        [
            A.OneOf([
                A.Affine(
                    scale=(0.88, 1.12),
                    translate_percent=(-0.06, 0.06),
                    rotate=(-8, 8),
                    shear=(-6, 6),
                    p=1.0,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.06,
                    scale_limit=0.12,
                    rotate_limit=8,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0,
                ),
            ], p=0.85),

            # Flip poziomy zwykle OK dla narzędzi
            A.HorizontalFlip(p=0.5),

            # Bardzo mała perspektywa, by nie niszczyć boxów
            A.Perspective(scale=(0.02, 0.05), keep_size=True, pad_mode=cv2.BORDER_REFLECT_101, p=0.15),

            # Fotometria: jedna z mocniejszych zmian + drobne HSV
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            ], p=0.9),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=18, val_shift_limit=12, p=0.5),
            A.RGBShift(r_shift_limit=8, g_shift_limit=8, b_shift_limit=8, p=0.2),

            # Szumy i rozmycia jak z realnej kamery
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.4),

            # Spadek rozdzielczości + kompresja JPEG
            A.Downscale(scale_min=0.85, scale_max=0.95, p=0.20),
            A.ImageCompression(quality_lower=40, quality_upper=85, p=0.35),

            # Kontrast w cieniach i artefakty oświetlenia
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.15),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.12),
            A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.5), angle_lower=0.25, p=0.05),

            # Dziury/zasłonięcia – delikatniej (ważne dla małych obiektów)
            A.CoarseDropout(
                max_holes=4,
                hole_height_range=(0.03, 0.12),
                hole_width_range=(0.03, 0.12),
                fill_value=0,
                p=0.25,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.15,  # mniejsze niż 0.2, by nie kasować małych obiektów
        ),
    )


def apply_aug(img_rgb: np.ndarray,
              boxes_xywh: List[Tuple[float, float, float, float]],
              class_labels: List[int],
              aug: A.Compose):
    transformed = aug(image=img_rgb, bboxes=boxes_xywh, class_labels=class_labels)
    return transformed["image"], transformed["bboxes"], transformed["class_labels"]


# ---------------------------- Główna logika ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Zbalansuj YOLOv8 przez augmentację klas oraz negatywów (profil indoor).")
    ap.add_argument("--dataset", required=True, type=Path, help="Katalog datasetu (z data.yaml).")
    ap.add_argument("--split", default="train", choices=["train", "valid", "val", "test"],
                    help="Który split augmentować (zwykle train).")
    ap.add_argument("--target", default="max",
                    help="Cel instancji na klasę: 'max' (do maximum z klas) lub liczba (np. 2500).")
    ap.add_argument("--max-aug-per-image", type=int, default=3,
                    help="Maksymalna liczba augmentów z jednego obrazu na jedną klasę.")
    ap.add_argument("--neg-mult", type=int, default=0,
                    help="Ile augmentów wykonać dla KAŻDEGO negatywu (pusta etykieta). 0 = wyłączone.")
    ap.add_argument("--jpeg-quality", type=int, default=95, help="Jakość JPEG przy zapisie.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Losowość
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        cv2.setRNGSeed(args.seed)
    except Exception:
        pass

    # Ścieżki
    ds = args.dataset
    data = read_yaml(ds / "data.yaml")
    names = load_class_names(data)
    split_name = "val" if args.split == "valid" else args.split

    images_dir = ds / split_name / "images"
    labels_dir = ds / split_name / "labels"
    assert images_dir.exists() and labels_dir.exists(), f"Brak {images_dir} lub {labels_dir}"

    # Statystyki klas
    counts, file_to_classes = count_instances(labels_dir)
    n_cls = len(names)

    print("Aktualne instancje per klasa:")
    for i, n in enumerate(names):
        print(f"  {i:>2} {n:<16} : {counts.get(i, 0)}")

    # Ustal target
    target = max(counts.get(i, 0) for i in range(n_cls)) if str(args.target).lower() == "max" else int(args.target)
    print(f"\nCel instancji na klasę: {target}")

    # Mapowanie: klasa -> lista plików etykiet, w których występuje
    class_to_files: dict[int, List[Path]] = defaultdict(list)
    for lbl_file, clset in file_to_classes.items():
        for c in clset:
            class_to_files[c].append(lbl_file)

    aug = build_augment_pipeline()
    usage_counter = defaultdict(int)

    # ---------------------- Augmentacja NEGATYWÓW ----------------------
    neg_created = 0
    if args.neg_mult > 0:
        neg_files = [p for p in labels_dir.rglob("*.txt") if len(yolo_txt_to_boxes(p)) == 0]
        print(f"\nNegatywy (puste etykiety) znalezione: {len(neg_files)}")
        for txt in tqdm(neg_files, desc="augment negatives", unit="img"):
            img_path = find_image_for_label(labels_dir, images_dir, txt)
            if not img_path:
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            for _ in range(args.neg_mult):
                t = aug(image=img_rgb, bboxes=[], class_labels=[])
                img_aug_rgb = t["image"]
                img_aug_bgr = cv2.cvtColor(img_aug_rgb, cv2.COLOR_RGB2BGR)

                uid = random.randint(0, 1_000_000)
                out_stem = f"{img_path.stem}_negaug{uid}"
                img_out = images_dir / f"{out_stem}.jpg"
                lbl_out = labels_dir / f"{out_stem}.txt"

                cv2.imwrite(str(img_out), img_aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                lbl_out.write_text("", encoding="utf-8")  # nadal NEGATYW
                neg_created += 1
        print(f"Utworzono nowych negatywów: {neg_created}")

    # ------------------ Augmentacja klas poniżej targetu ------------------
    total_new_images = Counter()
    total_new_instances = Counter()

    for cls_id in range(n_cls):
        cur = counts.get(cls_id, 0)
        if cur >= target:
            continue
        need = target - cur
        files = class_to_files.get(cls_id, [])
        if not files:
            print(f"  [!] Klasa '{names[cls_id]}' nie występuje w ogóle — pomijam.")
            continue

        print(f"\nAugmentuję klasę '{names[cls_id]}' ({cur} -> {target}, potrzebuję +{need})")
        pbar = tqdm(total=need, desc=f"augment {names[cls_id]}", unit="inst")
        i_try = 0

        while need > 0 and i_try < need * 20:
            i_try += 1
            lbl_path = random.choice(files)
            stem = lbl_path.stem

            # limit augmentów z danego obrazu dla tej klasy
            if usage_counter[(cls_id, stem)] >= args.max_aug_per_image:
                continue

            img_path = find_image_for_label(labels_dir, images_dir, lbl_path)
            if img_path is None:
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            boxes_all = yolo_txt_to_boxes(lbl_path)
            if not boxes_all:
                continue

            class_labels = [b[0] for b in boxes_all]
            bboxes_xywh = [(b[1], b[2], b[3], b[4]) for b in boxes_all]

            try:
                img_aug_rgb, b_aug, l_aug = apply_aug(img_rgb, bboxes_xywh, class_labels, aug)
            except Exception:
                continue

            # Musi zawierać daną klasę
            if not b_aug or (cls_id not in l_aug):
                continue

            # Sanity-check bboxów (clamp, min size)
            new_boxes: List[Tuple[int, float, float, float, float]] = []
            for (x, y, w, h), c in zip(b_aug, l_aug):
                x = float(min(max(x, 0.0), 1.0))
                y = float(min(max(y, 0.0), 1.0))
                w = float(min(max(w, 0.0), 1.0))
                h = float(min(max(h, 0.0), 1.0))
                if w <= 1e-6 or h <= 1e-6:
                    continue
                new_boxes.append((int(c), x, y, w, h))
            if not new_boxes:
                continue

            # Zapis
            uid = random.randint(0, 1_000_000)
            out_stem = f"{stem}_aug{uid}"
            img_out = images_dir / f"{out_stem}.jpg"
            lbl_out = labels_dir / f"{out_stem}.txt"

            img_aug_bgr = cv2.cvtColor(img_aug_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_out), img_aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            lbl_out.write_text(boxes_to_yolo_txt(new_boxes) + "\n", encoding="utf-8")

            # Aktualizacje liczników
            add_for_cls = sum(1 for cc, *_ in new_boxes if cc == cls_id)
            counts[cls_id] += add_for_cls
            need -= add_for_cls
            usage_counter[(cls_id, stem)] += 1
            total_new_images[cls_id] += 1
            total_new_instances[cls_id] += add_for_cls
            pbar.update(add_for_cls)

        pbar.close()
        if counts[cls_id] < target:
            print(f"  [!] Nie osiągnięto targetu dla '{names[cls_id]}': {counts[cls_id]} / {target}")

    # Podsumowanie
    print("\nPo augmentacji instancje per klasa:")
    for i, n in enumerate(names):
        print(f"  {i:>2} {n:<16} : {counts.get(i, 0)}")

    if args.neg_mult > 0:
        print(f"\nUtworzono nowych negatywów (obrazów): {neg_created}")

    print("\nNowe obrazy / instancje per klasa (tylko z oversamplingu klas):")
    for i, n in enumerate(names):
        print(f"  {i:>2} {n:<16} : {total_new_images[i]} img, {total_new_instances[i]} inst.")


if __name__ == "__main__":
    main()
