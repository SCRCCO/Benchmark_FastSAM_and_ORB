#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark FastSAM (x & s) a varie risoluzioni + sweep ORB.
Esegue senza argomenti e produce UN SOLO CSV: benchmark_results.csv

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

# Download utils
import urllib.request
from urllib.error import URLError, HTTPError

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
    # Con ultralytics: FastSAMClass(path) funziona come previsto (user request)
    model = FastSAM(model_path)
    rows = []
    for size in sizes:
        for name, img in images:
            # warmup
            _ = model(
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
                _ = model(
                    img,
                    device=device,
                    retina_masks=retina_masks,
                    imgsz=size,
                    conf=conf,
                    iou=iou,
                )
                sync_device(device)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                rows.append(
                    {
                        # comuni
                        "test_type": "fastsam",
                        "framework": "FastSAM",
                        "variant": Path(model_path).name,
                        "device": device,
                        "image": name,
                        "run": r + 1,
                        "latency_ms": round(dt_ms, 3),
                        # fastsam-specific
                        "img_size": size,
                        "conf": conf,
                        "iou": iou,
                        "retina_masks": retina_masks,
                        # orb-specific (NaN)
                        "n_keypoints": np.nan,
                        "desc_shape_0": np.nan,
                        "desc_shape_1": np.nan,
                        "kp_per_ms": np.nan,
                        "nfeatures": np.nan,
                        "scaleFactor": np.nan,
                        "nlevels": np.nan,
                        "edgeThreshold": np.nan,
                        "WTA_K": np.nan,
                        "scoreType": np.nan,
                        "patchSize": np.nan,
                        "fastThreshold": np.nan,
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
        "nfeatures": [200, 500],
        "scaleFactor": [1.1, 1.2],
        "nlevels": [4, 8],
        "edgeThreshold": [15, 31],
        "WTA_K": [2],  # fisso (riduce combinazioni)
        "scoreType": [cv2.ORB_HARRIS_SCORE],  # fisso (stabile)
        "patchSize": [31],
        "fastThreshold": [5, 10],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if len(combos) > max_combos:
        combos = combos[:max_combos]

    rows = []
    _ = cv2.ORB_create()  # warmup default

    total_iters = len(combos) * repeats
    pbar = tqdm(total=total_iters, desc=f"ORB {image_name}") if _HAS_TQDM else None

    for vals in combos:
        params = dict(zip(keys, vals))
        try:
            orb = cv2.ORB_create(**params)
        except Exception:
            continue

        _ = orb.detect(gray, None)  # warmup

        for r in range(repeats):
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
                    "device": "cpu",  # ORB è CPU, lascia così anche su GPU
                    "image": image_name,
                    "run": r + 1,
                    "latency_ms": round(dt_ms, 3),
                    "img_size": np.nan,
                    "conf": np.nan,
                    "iou": np.nan,
                    "retina_masks": np.nan,
                    "n_keypoints": n_kp,
                    "desc_shape_0": dshape0,
                    "desc_shape_1": dshape1,
                    "kp_per_ms": (n_kp / max(dt_ms, 1e-6)),
                    **params,
                }
            )

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()
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

    dfs = []

    # FastSAM-x

    print(f"-- Benchmark FastSAM-x.pt")
    try:
        dfs.append(
            run_fastsam_bench("FastSAM-x.pt", device, sizes, images, repeats=repeats)
        )
    except Exception as e:
        print(f"[ERRORE] FastSAM-x: {e}", file=sys.stderr)

    # FastSAM-s

    print(f"-- Benchmark FastSAM-s.pt")
    try:
        dfs.append(
            run_fastsam_bench("FastSAM-s.pt", device, sizes, images, repeats=repeats)
        )
    except Exception as e:
        print(f"[ERRORE] FastSAM-s: {e}", file=sys.stderr)

    print("-- ORB sweep solo sulla prima")
    try:
        for name, im in images_detection[:1]:
            dfs.append(run_orb_sweep(im, name, repeats=3))
    except Exception as e:
        print(f"[ERRORE] ORB: {e}", file=sys.stderr)

    if not dfs:
        print(
            "Nessun risultato generato. Controlla modelli/immagini e dipendenze.",
            file=sys.stderr,
        )
        sys.exit(2)

    final_df = pd.concat(dfs, ignore_index=True)

    # ordina per tipo e poi per latenza
    final_df = final_df.sort_values(
        ["test_type", "image", "variant", "img_size", "run"],
        na_position="last",
        kind="mergesort",
    )

    out_path = "benchmark_results.csv"
    save_csv(final_df, out_path)

    with pd.option_context("display.max_rows", 20, "display.max_columns", None):
        print("\nAnteprima risultati:")
        print(final_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
