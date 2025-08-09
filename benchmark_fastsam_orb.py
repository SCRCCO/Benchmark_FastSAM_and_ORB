#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark FastSAM (x & s) a varie risoluzioni + sweep ORB.
Esegue senza argomenti e produce DUE CSV:
- fastsam_benchmark.csv
- orb_benchmark.csv

- Immagini: ./images/*.jpg|png|bmp|tif (ne usa 8; se >8 prende le prime 8).
- Immagini ORB: ./images_detection/* (se vuota, usa la prima di ./images)
- Pesi: FastSAM-x.pt e FastSAM-s.pt nella cartella corrente.
  Se mancanti, prova a SCARICARLI automaticamente da una lista di URL.
- Device: auto (cuda -> mps -> cpu).

CSV: include tutte le colonne; per righe FastSAM i campi ORB sono NaN e viceversa.
"""

import os, sys, glob, time, itertools, platform
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import FastSAM

# Torch
try:
    import torch
except Exception:
    print(
        "ERRORE: PyTorch non disponibile. Installa una build adatta alla tua piattaforma.",
        file=sys.stderr,
    )
    raise

# OpenCV per ORB
import cv2

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


# ----------------------- Utility -----------------------
def pick_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_device(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def list_images(folder="images"):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(str(Path(folder) / ext)))
    return sorted(paths)


def load_images(paths):
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Impossibile leggere immagine: {p}")
        imgs.append((Path(p).name, img))
    return imgs


def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"[OK] Salvato: {path} ({len(df)} righe)")


# ----------------------- FastSAM benchmark -----------------------
def run_fastsam_bench(
    model_path, device, sizes, images, repeats=5, conf=0.4, iou=0.9, retina_masks=True
):
    model = FastSAM(model_path)
    rows = []
    for size in sizes:
        for name, img in images:
            # warmup
            result = model(
                img,
                device=device,
                retina_masks=retina_masks,
                imgsz=size,
                conf=conf,
                iou=iou,
            )
            sync_device(device)
            for r in range(repeats):
                t0 = time.perf_counter()
                result = model(
                    img,
                    device=device,
                    retina_masks=retina_masks,
                    imgsz=size,
                    conf=conf,
                    iou=iou,
                )
                sync_device(device)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                n_detections = (
                    len(result[0].masks.data)
                    if hasattr(result[0], "masks") and hasattr(result[0].masks, "data")
                    else 0
                )
                rows.append(
                    {
                        "test_type": "fastsam",
                        "framework": "FastSAM",
                        "variant": Path(model_path).name,
                        "device": device,
                        "image": name,
                        "run": r + 1,
                        "latency_ms": round(dt_ms, 3),
                        "img_size": size,
                        "conf": conf,
                        "iou": iou,
                        "retina_masks": retina_masks,
                        "n_detections": n_detections,
                    }
                )
    return pd.DataFrame(rows)


# ----------------------- ORB sweep -----------------------
def run_orb_sweep(image, image_name, repeats=3, max_side=1024, max_combos=200):
    try:
        from tqdm import tqdm

        _HAS_TQDM = True
    except Exception:
        _HAS_TQDM = False

    def _downscale_max(gray, max_side=1024):
        h, w = gray.shape[:2]
        m = max(h, w)
        if m <= max_side:
            return gray
        scale = max_side / float(m)
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

    cv2.setNumThreads(1)  # evita overhead/lock strani su sistemi piccoli
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = _downscale_max(gray, max_side=max_side)

    grid = {
        "nfeatures": [100, 500, 1000, 2500, 5000],
        "scaleFactor": [1.1, 1.2, 1.3, 1.5],
        "nlevels": [4, 6, 8, 12],
        "edgeThreshold": [10, 20, 31, 50],
        "firstLevel": [0],  # fisso
        "WTA_K": [2, 3, 4],
        "scoreType": [cv2.ORB_HARRIS_SCORE, cv2.ORB_FAST_SCORE],
        "patchSize": [31, 21, 41],
        "fastThreshold": [5, 10, 20, 30],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if len(combos) > max_combos:
        step = len(combos) // max_combos
        combos = combos[::step][:max_combos]

    rows = []

    total_iters = len(combos) * repeats
    pbar = tqdm(total=total_iters, desc=f"ORB {image_name}") if _HAS_TQDM else None

    print(f"Testando {len(combos)} combinazioni di parametri ORB...")

    for combo_idx, vals in enumerate(combos):
        params = dict(zip(keys, vals))

        # Debug: stampa ogni 50 combinazioni per verificare che i parametri cambino
        if combo_idx % 50 == 0:
            print(
                f"Combo {combo_idx}: nfeatures={params['nfeatures']}, scaleFactor={params['scaleFactor']}, edgeThreshold={params['edgeThreshold']}"
            )

        try:
            # Crea SEMPRE un nuovo oggetto ORB per ogni combinazione di parametri
            orb = cv2.ORB_create(
                nfeatures=params["nfeatures"],
                scaleFactor=params["scaleFactor"],
                nlevels=params["nlevels"],
                edgeThreshold=params["edgeThreshold"],
                firstLevel=params["firstLevel"],
                WTA_K=params["WTA_K"],
                scoreType=params["scoreType"],
                patchSize=params["patchSize"],
                fastThreshold=params["fastThreshold"],
            )
        except Exception as e:
            print(f"Errore creazione ORB con params {params}: {e}")
            continue

        # Warmup con il nuovo oggetto ORB
        try:
            _ = orb.detectAndCompute(gray, None)
        except Exception:
            continue

        for r in range(repeats):
            try:
                t0 = time.perf_counter()
                kp, des = orb.detectAndCompute(gray, None)
                dt_ms = (time.perf_counter() - t0) * 1000.0

                n_kp = 0 if kp is None else len(kp)
                dshape0, dshape1 = des.shape if des is not None else (0, 0)

                rows.append(
                    {
                        "test_type": "orb",
                        "framework": "OpenCV",
                        "variant": "ORB",
                        "device": "cpu",
                        "image": image_name,
                        "run": r + 1,
                        "latency_ms": round(dt_ms, 3),
                        # ORB specifici
                        "n_keypoints": n_kp,
                        "desc_shape_0": dshape0,
                        "desc_shape_1": dshape1,
                        "kp_per_ms": round(n_kp / max(dt_ms, 1e-6), 2),
                        # Parametri ORB
                        "nfeatures": params["nfeatures"],
                        "scaleFactor": params["scaleFactor"],
                        "nlevels": params["nlevels"],
                        "edgeThreshold": params["edgeThreshold"],
                        "firstLevel": params["firstLevel"],
                        "WTA_K": params["WTA_K"],
                        "scoreType": params["scoreType"],
                        "patchSize": params["patchSize"],
                        "fastThreshold": params["fastThreshold"],
                    }
                )

            except Exception as e:
                print(f"Errore durante detection: {e}")
                continue

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    print(f"Completate {len(rows)} misurazioni ORB")
    return pd.DataFrame(rows)


# ----------------------- Main -----------------------
def main():
    print("== Benchmark FastSAM + ORB ==")
    print(
        f"Python: {platform.python_version()} | torch: {torch.__version__} | opencv: {cv2.__version__}"
    )

    # immagini (max 8)
    img_paths = list_images("images")
    if len(img_paths) < 3:
        print("ERRORE: servono almeno 3 immagini in ./images", file=sys.stderr)
        sys.exit(1)
    img_paths = img_paths[:8]
    images = load_images(img_paths)

    # immagini ORB (fallback alla prima di images)
    img_det_paths = list_images("images_detection")
    if not img_det_paths:
        img_det_paths = [img_paths[0]]
    images_detection = load_images(img_det_paths)

    # device
    device = pick_device()
    print(f"Dispositivo scelto: {device}")

    sizes = [512, 576, 640, 704, 768, 832, 896, 960, 1024]
    repeats = 5

    # ========== FASTSAM ==========
    dfs_fastsam = []

    print("-- Benchmark FastSAM-x.pt")
    try:
        dfs_fastsam.append(
            run_fastsam_bench("FastSAM-x.pt", device, sizes, images, repeats=repeats)
        )
    except Exception as e:
        print(f"[ERRORE] FastSAM-x: {e}", file=sys.stderr)

    print("-- Benchmark FastSAM-s.pt")
    try:
        dfs_fastsam.append(
            run_fastsam_bench("FastSAM-s.pt", device, sizes, images, repeats=repeats)
        )
    except Exception as e:
        print(f"[ERRORE] FastSAM-s: {e}", file=sys.stderr)

    if dfs_fastsam:
        df_fastsam = pd.concat(dfs_fastsam, ignore_index=True)
        df_fastsam = df_fastsam.sort_values(
            ["variant", "image", "img_size", "run"], kind="mergesort"
        )
        save_csv(df_fastsam, "fastsam_benchmark.csv")
    else:
        print("[WARN] Nessun risultato FastSAM generato.")

    # ========== ORB ==========
    dfs_orb = []
    print("-- ORB sweep")
    try:
        for name, im in images_detection:
            print(f"Processando immagine: {name}")
            dfs_orb.append(
                run_orb_sweep(im, name, repeats=3, max_side=1024, max_combos=200)
            )
    except Exception as e:
        print(f"[ERRORE] ORB: {e}", file=sys.stderr)

    if dfs_orb:
        df_orb = pd.concat(dfs_orb, ignore_index=True)
        df_orb = df_orb.sort_values(
            ["image", "latency_ms", "n_keypoints"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        save_csv(df_orb, "orb_benchmark.csv")

        # Analisi veloce per verificare la varianza
        print(f"\nStatistiche ORB:")
        print(
            f"Range latenza: {df_orb['latency_ms'].min():.2f} - {df_orb['latency_ms'].max():.2f} ms"
        )
        print(
            f"Range keypoints: {df_orb['n_keypoints'].min()} - {df_orb['n_keypoints'].max()}"
        )
        print(f"Varianza latenza: {df_orb['latency_ms'].var():.2f}")
        print(f"Varianza keypoints: {df_orb['n_keypoints'].var():.2f}")
    else:
        print("[WARN] Nessun risultato ORB generato.")

    # Stampa breve anteprima
    if dfs_fastsam:
        print("\nAnteprima FASTSAM:")
        with pd.option_context("display.max_rows", 10, "display.max_columns", None):
            print(df_fastsam.head(10).to_string(index=False))
    if dfs_orb:
        print("\nAnteprima ORB:")
        with pd.option_context("display.max_rows", 10, "display.max_columns", None):
            print(df_orb.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
