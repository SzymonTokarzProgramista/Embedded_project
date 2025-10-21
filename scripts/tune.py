import argparse
import os
import yaml
from datetime import datetime
from typing import Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from ultralytics import YOLO
from ultralytics.cfg import get_cfg


# ------------------------------
# Helper: evaluate a model on a dataset and return mAP50-95
# ------------------------------
def eval_map(model: YOLO, data_yaml: str) -> float:
    results = model.val(data=data_yaml, verbose=False)
    # Ultralytics returns a Results object with .box.map (mAP50-95)
    try:
        return float(results.box.map)
    except Exception:
        # Fallback in case API changes
        return float(getattr(results, "map", 0.0))


# ------------------------------
# Optuna objective with anti-forgetting regularization
# ------------------------------
def objective(trial: optuna.Trial, args: argparse.Namespace, baseline_general_map: Optional[float]) -> float:
    cfg = get_cfg()

    # Core training setup (shorter, cheaper HPO)
    cfg.data     = args.data
    cfg.epochs   = args.search_epochs
    cfg.patience = 3
    cfg.fraction = args.hpo_fraction
    cfg.device   = args.device
    cfg.batch    = 0
    cfg.workers  = args.workers

    # --- Hyperparams to search ---
    cfg.imgsz        = trial.suggest_categorical("imgsz", [416, 512, 640, 768, 960])
    cfg.lr0          = trial.suggest_float("lr0", 5e-5, 2e-3, log=True)  # lower range for safer fine-tuning
    cfg.lrf          = trial.suggest_float("lrf", 0.05, 0.5)             # cosine final LR factor
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # Augmentations (kept modest to avoid overfitting tiny target domains)
    cfg.scale       = trial.suggest_float("scale", 0.4, 0.9)
    cfg.mosaic      = trial.suggest_float("mosaic", 0.05, 0.25)
    cfg.mixup       = trial.suggest_float("mixup", 0.0, 0.12)
    cfg.copy_paste  = trial.suggest_float("copy_paste", 0.05, 0.25)

    # --- Sensible constants ---
    cfg.momentum      = 0.9
    cfg.dropout       = 0.0
    cfg.cos_lr        = True
    cfg.amp           = True
    cfg.warmup_epochs = 3
    cfg.freeze = trial.suggest_categorical("freeze", list(range(0, args.freeze_max + 1))) \
                 if args.freeze_max > 0 else 0
    cfg.deterministic = True
    cfg.verbose       = False

    # Housekeeping
    cfg.project = os.path.join(args.project, "optuna")
    cfg.name    = f"trial_{trial.number}"

    train_args = {k: v for k, v in vars(cfg).items() if k != "model"}

    # Train from the general model weights
    model = YOLO(args.weights)
    results = model.train(**train_args)

    # Evaluate on target (test environment) and general validation
    target_map = float(results.box.map)

    if args.general_data:
        general_map = eval_map(model, args.general_data)
    else:
        general_map = 0.0

    # Anti-forgetting objective: reward target mAP, penalize drop vs. baseline general mAP
    # drop = max(0, baseline - new_general)
    if baseline_general_map is not None and args.general_data:
        drop = max(0.0, baseline_general_map - general_map)
    else:
        drop = 0.0

    # Higher alpha favors target gain, higher beta punishes forgetting more.
    score = args.alpha * target_map - args.beta * drop

    # Optuna minimizes, so return 1 - score (clipped so it's well-behaved)
    objective_value = 1.0 - max(0.0, min(1.0, score))

    # Attach useful info
    trial.set_user_attr("target_map", target_map)
    trial.set_user_attr("general_map", general_map)
    trial.set_user_attr("drop_vs_baseline", drop)

    return objective_value


# ------------------------------
# Main
# ------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="YOLO Optuna fine-tune with anti-forgetting (domain adaptation)")
    p.add_argument("--data", required=True, help="data.yaml for TARGET (test environment)")
    p.add_argument("--weights", default="yolov8n.pt", help="pretrained GENERAL model weights or checkpoint")
    p.add_argument("--project", default="runs", help="output root dir")

    # HPO / training schedule
    p.add_argument("--trials", type=int, default=1000, help="Optuna trials")
    p.add_argument("--search-epochs", type=int, default=200, help="epochs per HPO trial")
    p.add_argument("--train-epochs", type=int, default=500, help="epochs final training")
    p.add_argument("--patience", type=int, default=40, help="early-stop patience final")

    # System
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--device", default="0", help='e.g. "0" (GPU) or "cpu"')
    p.add_argument("--workers", type=int, default=8)

    # Data fractions
    p.add_argument("--hpo-fraction", type=float, default=0.5, help="fraction of TARGET data to use in HPO (0-1)")

    # Freezing during HPO (helps preserve generalization)
    p.add_argument("--freeze-max", type=int, default=8, help="max layers to freeze in HPO (0 disables search)")

    # Generalization preservation controls
    p.add_argument("--general-data", default=None, help="data.yaml for GENERAL validation (unchanged domain)")
    p.add_argument("--alpha", type=float, default=0.85, help="weight for TARGET mAP in objective (0-1)")
    p.add_argument("--beta", type=float, default=0.50, help="weight for penalty of GENERAL mAP drop")
    p.add_argument("--max-general-drop", type=float, default=0.02, help="tolerated absolute drop in GENERAL mAP (e.g., 0.02 = 2pp)")

    args = p.parse_args()

    os.makedirs(args.project, exist_ok=True)

    # Baseline GENERAL mAP (how good the general model is before adaptation)
    baseline_general_map: Optional[float] = None
    if args.general_data:
        print("\n📏 Evaluating baseline GENERAL mAP (before fine-tuning)…")
        baseline_general_map = eval_map(YOLO(args.weights), args.general_data)
        print(f"   Baseline GENERAL mAP50-95: {baseline_general_map:.4f}")

    # ---------------- HPO ----------------
    study_name = f"yolo_hpo_preserve_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=TPESampler(seed=args.seed, n_startup_trials=8),
        pruner=MedianPruner(n_warmup_steps=3),
    )

    print("\n🔎 Starting HPO with anti-forgetting objective…")
    study.optimize(lambda t: objective(t, args, baseline_general_map), n_trials=args.trials)

    best_trial = study.best_trial
    best_params = best_trial.params

    print("\n🏆 Najlepsze hiperparametry (HPO):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print("\n📊 Metryki najlepszego triala:")
    print(f"  TARGET mAP50-95: {best_trial.user_attrs.get('target_map', float('nan')):.4f}")
    if args.general_data:
        print(f"  GENERAL mAP50-95: {best_trial.user_attrs.get('general_map', float('nan')):.4f}")
        print(f"  Drop vs. baseline: {best_trial.user_attrs.get('drop_vs_baseline', float('nan')):.4f}")

    hp_path = os.path.join(args.project, "hpo_best_params.yaml")
    with open(hp_path, "w", encoding="utf-8") as f:
        yaml.dump(best_params, f, sort_keys=False)
    print(f"\n💾 Zapisano {hp_path}\n")

    # ---------------- FINAL TRAIN on FULL TARGET data ----------------
    # Two-phase fine-tune: (1) more frozen, (2) unfreeze for a short polish
    print("🚀 Start final fine-tuning (two-phase)…")

    # Common cfg for both phases
    base_cfg = get_cfg()
    base_cfg.data     = args.data
    base_cfg.project  = os.path.join(args.project, "train")
    base_cfg.name     = "final_best"
    base_cfg.warmup_epochs = 3
    base_cfg.cos_lr   = True
    base_cfg.amp      = True
    base_cfg.batch    = 0
    base_cfg.device   = args.device
    base_cfg.workers  = args.workers
    base_cfg.deterministic = True
    base_cfg.verbose  = True

    # Apply tuned hparams
    for k in ["imgsz", "lr0", "lrf", "weight_decay", "scale", "mosaic", "mixup", "copy_paste"]:
        if k in best_params:
            setattr(base_cfg, k, best_params[k])

    # Phase 1: conservative head-only (or lightly unfrozen) fine-tune
    cfg1 = get_cfg()
    cfg1.__dict__.update(base_cfg.__dict__)
    cfg1.epochs   = max(5, int(args.train_epochs * 0.7))
    cfg1.patience = args.patience
    cfg1.freeze   = min(8, max(1, best_params.get("freeze", 0)))  # keep most of the backbone fixed

    print("\n🧊 Phase 1: partially frozen fine-tune…")
    model = YOLO(args.weights)
    model.train(**{k: v for k, v in vars(cfg1).items() if k != "model"})

    # Phase 2: brief unfreeze & polish with smaller LR
    cfg2 = get_cfg()
    cfg2.__dict__.update(base_cfg.__dict__)
    cfg2.epochs   = max(3, args.train_epochs - cfg1.epochs)
    cfg2.patience = max(5, int(args.patience * 0.5))
    cfg2.freeze   = 0
    # reduce LR for polish (safer for generalization)
    if hasattr(cfg2, "lr0"):
        cfg2.lr0 = float(cfg2.lr0) * 0.5

    print("\n🫧 Phase 2: unfreeze & low-LR polish…")
    # Continue from Phase 1 checkpoint (best from last train run stored in model)
    model.train(**{k: v for k, v in vars(cfg2).items() if k != "model"})

    # ---------------- Final evaluation ----------------
    print("\n✅ Final evaluation…")
    target_final_map = eval_map(model, args.data)
    print(f"TARGET mAP50-95: {target_final_map:.4f}")

    if args.general_data:
        general_final_map = eval_map(model, args.general_data)
        print(f"GENERAL mAP50-95: {general_final_map:.4f}")
        if baseline_general_map is not None:
            drop = baseline_general_map - general_final_map
            print(f"Drop vs. baseline (GENERAL): {drop:+.4f}")
            if drop > args.max_general_drop:
                print("⚠️  Uwaga: spadek dokładności na zbiorze ogólnym przekracza próg. "
                      "Rozważ mniejszy LR, większe zamrożenie warstw lub dodanie odrobiny danych ogólnych do fine-tuningu.")

    print("\n🎯 Gotowe! Sprawdź wyniki w katalogu:")
    print(f"   - HPO:     {os.path.join(args.project, 'optuna')}")
    print(f"   - Training {os.path.join(args.project, 'train')}")


if __name__ == "__main__":
    main()
