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
import math

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


# ----------------------- ORB orientation+rotation benchmark -----------------------
def _downscale_img_max(img, max_side=1024):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def _read_gray_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def _rotate_center_img(gray, angle_deg):
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def run_orb_pose_orientation_bench(
    ref_front_path="Coop_fronte_texture_fronte.png",
    ref_back_path="Coop_retro_texture_retro.png",
    true_rotation_deg=45.0,
    repeats=5,
    nfeatures_list=(500, 1000, 2000, 3000),
    base_rotations=(0, 90, 180, 270),
    max_side=1024,
    out_csv="orb_pose_benchmark.csv",
):
    """
    Genera due test:
      - FRONT ruotato di true_rotation_deg
      - BACK  ruotato di true_rotation_deg
    Per ogni test, stima ORIENTAMENTO (front/back) e ANGOLO provando entrambe le reference
    e tutte le base_rotations, poi sceglie la combinazione con più good matches.
    Ripete per ogni nfeatures e salva CSV, includendo i TEMPI principali.
    """
    print(
        f"-- ORB pose benchmark: front={ref_front_path}, back={ref_back_path}, rot={true_rotation_deg}°, repeats={repeats}"
    )

    # Carica le reference in gray
    _, front_gray = _read_gray_img(ref_front_path)
    _, back_gray = _read_gray_img(ref_back_path)

    # Downscale (opzionale)
    if max_side:
        front_gray = _downscale_img_max(front_gray, max_side)
        back_gray = _downscale_img_max(back_gray, max_side)

    # Crea i due test (front/back ruotati)
    tests = [
        ("front", _rotate_center_img(front_gray, true_rotation_deg)),
        ("back", _rotate_center_img(back_gray, true_rotation_deg)),
    ]
    refs = {"front": front_gray, "back": back_gray}

    # Progress bar
    try:
        from tqdm import tqdm

        use_pbar = True
    except Exception:
        use_pbar = False
    total_runs = len(tests) * len(nfeatures_list) * repeats
    pbar = tqdm(total=total_runs, desc="ORB pose") if use_pbar else None

    rows = []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    for gt_orientation, test_gray in tests:
        for nf in nfeatures_list:
            for r in range(1, repeats + 1):
                t_run0 = time.perf_counter()

                # 1) Crea ORB (tempo creazione)
                t0 = time.perf_counter()
                orb = cv2.ORB_create(
                    nfeatures=nf,
                    scaleFactor=1.2,
                    nlevels=8,
                    edgeThreshold=31,
                    WTA_K=2,
                    scoreType=cv2.ORB_HARRIS_SCORE,
                    patchSize=31,
                    fastThreshold=10,
                )
                time_orb_create_ms = (time.perf_counter() - t0) * 1000.0

                # 2) Feature del TEST (una volta per run)
                t0 = time.perf_counter()
                test_kp, test_des = orb.detectAndCompute(test_gray, None)
                time_test_detect_ms = (time.perf_counter() - t0) * 1000.0

                # 3) Ricerca migliore combinazione (cronometrata)
                t_search0 = time.perf_counter()
                best = None  # (good, pred_orient, total_deg, base_rot, fine_deg, inl,
                #  t_ref_rotate, t_ref_detect, t_match, t_est)
                candidates = 0

                for orient, ref_gray in refs.items():
                    for base in base_rotations:
                        # 3a) Rotazione reference
                        t0 = time.perf_counter()
                        if base == 0:
                            ref_rot = ref_gray
                        elif base == 90:
                            ref_rot = cv2.rotate(ref_gray, cv2.ROTATE_90_CLOCKWISE)
                        elif base == 180:
                            ref_rot = cv2.rotate(ref_gray, cv2.ROTATE_180)
                        elif base == 270:
                            ref_rot = cv2.rotate(
                                ref_gray, cv2.ROTATE_90_COUNTERCLOCKWISE
                            )
                        else:
                            ref_rot = ref_gray
                        time_ref_rotate_ms = (time.perf_counter() - t0) * 1000.0

                        # 3b) Feature reference
                        t0 = time.perf_counter()
                        ref_kp, ref_des = orb.detectAndCompute(ref_rot, None)
                        time_ref_detect_ms = (time.perf_counter() - t0) * 1000.0

                        if (
                            (ref_des is None)
                            or (test_des is None)
                            or (len(ref_kp) < 10)
                            or (len(test_kp) < 10)
                        ):
                            candidates += 1
                            continue

                        # 3c) Matching + filtro distanza
                        t0 = time.perf_counter()
                        matches = bf.match(test_des, ref_des)
                        matches = sorted(matches, key=lambda x: x.distance)
                        good = [m for m in matches if m.distance < 60]
                        time_match_ms = (time.perf_counter() - t0) * 1000.0

                        if len(good) < 10:
                            candidates += 1
                            continue

                        # 3d) Stima affine (angolo fine)
                        t0 = time.perf_counter()
                        src_pts = np.float32(
                            [test_kp[m.queryIdx].pt for m in good]
                        ).reshape(-1, 1, 2)
                        dst_pts = np.float32(
                            [ref_kp[m.trainIdx].pt for m in good]
                        ).reshape(-1, 1, 2)
                        M, inliers = cv2.estimateAffinePartial2D(
                            src_pts,
                            dst_pts,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=3.0,
                            confidence=0.99,
                        )
                        time_estimate_ms = (time.perf_counter() - t0) * 1000.0

                        if M is None:
                            candidates += 1
                            continue

                        angle_rad = np.arctan2(M[0, 1], M[0, 0])
                        fine_deg = (np.degrees(angle_rad)) % 360.0
                        total_deg = (base - fine_deg) % 360.0
                        inl = (
                            int(inliers.sum()) if isinstance(inliers, np.ndarray) else 0
                        )

                        cand = (
                            len(good),
                            orient,
                            total_deg,
                            base,
                            fine_deg,
                            inl,
                            time_ref_rotate_ms,
                            time_ref_detect_ms,
                            time_match_ms,
                            time_estimate_ms,
                        )
                        candidates += 1

                        if (
                            (best is None)
                            or (len(good) > best[0])
                            or (len(good) == best[0] and inl > best[5])
                        ):
                            best = cand

                time_search_ms = (time.perf_counter() - t_search0) * 1000.0
                run_total_ms = (time.perf_counter() - t_run0) * 1000.0

                # 4) Estrai risultati finali
                if best is None:
                    pred_orient = "unknown"
                    est_deg = 0.0
                    base_rot = np.nan
                    fine_deg = np.nan
                    good = 0
                    inl = 0
                    best_t_ref_rot = best_t_ref_det = best_t_match = best_t_est = 0.0
                else:
                    (
                        good,
                        pred_orient,
                        est_deg,
                        base_rot,
                        fine_deg,
                        inl,
                        best_t_ref_rot,
                        best_t_ref_det,
                        best_t_match,
                        best_t_est,
                    ) = best

                err = abs((est_deg - true_rotation_deg + 180) % 360 - 180)
                orient_ok = int(pred_orient == gt_orientation)

                # 5) Salva riga con tempi principali
                rows.append(
                    {
                        "gt_orientation": gt_orientation,
                        "pred_orientation": pred_orient,
                        "orientation_correct": orient_ok,
                        "true_deg": true_rotation_deg,
                        "estimated_deg": round(est_deg, 3),
                        "abs_error_deg": round(err, 3),
                        "base_rotation": base_rot,
                        "fine_angle": (
                            round(fine_deg, 3) if not np.isnan(fine_deg) else np.nan
                        ),
                        "good_matches": good,
                        "inliers": inl,
                        "nfeatures": nf,
                        "max_side": max_side,
                        "run": r,
                        # TEMPI (ms)
                        "time_orb_create_ms": round(time_orb_create_ms, 3),
                        "time_test_detect_ms": round(time_test_detect_ms, 3),
                        "time_search_ms": round(time_search_ms, 3),
                        "candidates_evaluated": candidates,
                        # pipeline della MIGLIORE candidate
                        "best_time_ref_rotate_ms": round(best_t_ref_rot, 3),
                        "best_time_ref_detect_ms": round(best_t_ref_det, 3),
                        "best_time_match_ms": round(best_t_match, 3),
                        "best_time_estimate_ms": round(best_t_est, 3),
                        "best_time_pipeline_ms": round(
                            best_t_ref_rot + best_t_ref_det + best_t_match + best_t_est,
                            3,
                        ),
                        # totale
                        "run_total_ms": round(run_total_ms, 3),
                    }
                )

                if pbar:
                    pbar.update(1)

    if pbar:
        pbar.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["gt_orientation", "nfeatures", "run"], kind="mergesort")
        df.to_csv(out_csv, index=False)
        print(f"[OK] Salvato: {out_csv} ({len(df)} righe)")
    else:
        print("[WARN] Nessun risultato ORB pose generato.")
    return df


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

    # ========== ORB ORIENTATION+ROTATION (nuova tabella) ==========
    try:
        df_pose = run_orb_pose_orientation_bench(
            ref_front_path="images_detection/Coop_fronte_texture_fronte.png",
            ref_back_path="images_detection/Coop_retro_texture_retro.png",
            true_rotation_deg=45.0,
            repeats=5,  # run per ciascun nfeatures e ciascun test (front/back)
            nfeatures_list=(500, 1000, 2000, 3000),  # più keypoints
            base_rotations=(0, 90, 180, 270),
            max_side=1024,
            out_csv="orb_pose_benchmark.csv",
        )
    except Exception as e:
        print(f"[WARN] ORB pose benchmark saltato: {e}", file=sys.stderr)

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
