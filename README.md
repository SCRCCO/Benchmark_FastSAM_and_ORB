````markdown
# Benchmark FastSAM + ORB (Multipiattaforma)

Script di benchmark per:
- **FastSAM-x** e **FastSAM-s** a risoluzioni: `512, 576, 640, 704, 768, 832, 896, 960, 1024` su **3 immagini statiche**.
- **ORB (OpenCV)** con sweep di combinazioni parametriche, misurando **latenza**, **#keypoint** e **keypoints/ms**.

**Output:** un unico CSV `benchmark_results.csv` con **tutte** le colonne (i campi non pertinenti a un test sono `NaN`).

---

## Indice
- [Requisiti](#requisiti)
- [Piattaforme supportate](#piattaforme-supportate)
- [Setup ambiente](#setup-ambiente)
- [Installazione PyTorch](#installazione-pytorch)
- [Installazione dipendenze](#installazione-dipendenze)
- [Struttura progetto](#struttura-progetto)
- [Uso](#uso)
- [Download automatico dei pesi](#download-automatico-dei-pesi)
- [Schema CSV](#schema-csv)
- [Uso su Jetson (Nano/Orin/Xavier)](#uso-su-jetson-nanoorinxaiver)
- [Ottimizzazioni & consigli](#ottimizzazioni--consigli)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Crediti](#crediti)

---

## Requisiti

- **Python 3.9 – 3.11** (consigliato **3.10** per massima compatibilità).
- **PyTorch** installato correttamente per la tua piattaforma (CPU/CUDA/macOS Metal).
- **Connessione internet** se i pesi non sono presenti localmente (lo script li scarica in automatico).

> Lo script seleziona automaticamente il device: **CUDA → MPS (macOS) → CPU**.

---

## Piattaforme supportate
- **Windows 10/11** (x64)
- **macOS** (Intel e Apple Silicon M1/M2/M3)
- **Linux** (x64)
- **NVIDIA Jetson** (Nano/Orin/Xavier) — vedi sezione dedicata

---

## Setup ambiente

### Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
````

### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

---

## Installazione PyTorch

Scegli la build corretta (usa anche la guida su [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).

**CPU only:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 12.4 (GPU NVIDIA):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**macOS (Apple Silicon / Metal):**

```bash
pip install torch torchvision torchaudio
```

> Su Jetson NON usare questi comandi generici: vedi la sezione **Uso su Jetson**.

---

## Installazione dipendenze

```bash
pip install -r requirements.txt
```

Se su Linux/Jetson hai conflitti con OpenCV di sistema, usa:

```bash
pip install opencv-python-headless
```

---

## Struttura progetto

```
project/
├── benchmark_fastsam_orb.py     # Script principale (no-args)
├── requirements.txt
├── README.md
├── FastSAM-x.pt                  # (opzionale: lo script può scaricarli)
├── FastSAM-s.pt                  # (opzionale: lo script può scaricarli)
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── image3.jpg
```

> Metti almeno **3 immagini** in `images/` (jpg/png/bmp/tif). Se ce ne sono di più, usa le prime 3 in ordine alfabetico.

---

## Uso

Esegui senza argomenti:

```bash
python benchmark_fastsam_orb.py
```

* Device scelto automaticamente (CUDA/MPS/CPU).
* Se i pesi non sono presenti, lo script prova a **scaricarli** (vedi sotto).

---

## Download automatico dei pesi

Se `FastSAM-x.pt` / `FastSAM-s.pt` non sono nella cartella del progetto, lo script tenta il download da più URL (mirror).
Gli URL possono cambiare; in caso di fallimento, scarica manualmente e metti i file nella stessa cartella dello script.

Suggerimento: su macchine con poca RAM (es. Jetson Nano), **usa solo `FastSAM-s.pt`** (rinomina/ometti `FastSAM-x.pt`).

---

## Schema CSV

Colonne comuni:

* `test_type` — `"fastsam"` o `"orb"`
* `framework` — `"FastSAM"` / `"OpenCV"`
* `variant` — `FastSAM-x.pt`, `FastSAM-s.pt`, o `ORB`
* `device` — `cuda`, `mps`, `cpu`, …
* `image` — nome file immagine
* `run` — indice della run
* `latency_ms` — latenza in millisecondi

**Specifiche FastSAM:**

* `img_size` — lato d’ingresso (p.es. 768)
* `conf`, `iou`, `retina_masks` — parametri di inferenza

**Specifiche ORB:**

* `n_keypoints`, `desc_shape_0`, `desc_shape_1`, `kp_per_ms`
* `nfeatures`, `scaleFactor`, `nlevels`, `edgeThreshold`, `WTA_K`, `scoreType`, `patchSize`, `fastThreshold`

I campi non pertinenti a un test sono `NaN`.

---

## Uso su Jetson (Nano/Orin/Xavier)

I dispositivi Jetson hanno RAM/CPU limitate; ecco il percorso consigliato.

### 1) Ambiente & dipendenze

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel
pip install opencv-python-headless numpy pandas pillow tqdm
pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git
```

### 2) PyTorch per Jetson

Installa la **ruota compatibile** con la tua versione di JetPack/CUDA (gli URL cambiano nel tempo: verifica sul forum NVIDIA “PyTorch for Jetson”).

**Esempio — JetPack 4.6.1, Python 3.8:**

```bash
wget https://nvidia.box.com/shared/static/3nm3gxxwy8f8q4nwd6kgtl4p3yr3zd8i.whl -O torch-1.10.0-cp38-cp38-linux_aarch64.whl
pip install torch-1.10.0-cp38-cp38-linux_aarch64.whl
pip install torchvision==0.11.1
```

**Esempio — JetPack 5.1, Python 3.8:**

```bash
wget https://nvidia.box.com/shared/static/bz5lxprp6k25lhk1pnc6u9h7v4rfv74i.whl -O torch-2.1.0-cp38-cp38-linux_aarch64.whl
pip install torch-2.1.0-cp38-cp38-linux_aarch64.whl
pip install torchvision==0.16.0
```

> Se il download è lento/instabile, valuta di scaricare su PC e copiare via scp/usb.

### 3) Modalità “Jetson Safe” (consigliata)

Per evitare OOM e ridurre i tempi:

* **Usa solo `FastSAM-s.pt`** (rimuovi/renomina `FastSAM-x.pt`)
* **Riduci risoluzioni**: `512, 576, 640, 704, 768`
* **Riduci ripetizioni**: da `5` a `3`

> Lo script attuale non ha flag: per applicare queste impostazioni, apri `benchmark_fastsam_orb.py` e modifica:
>
> * `sizes = [512, 576, 640, 704, 768]`
> * `repeats = 3`
> * assicurati che sia presente solo `FastSAM-s.pt` nella cartella

### 4) Esecuzione

```bash
python benchmark_fastsam_orb.py
```

---

## Ottimizzazioni & consigli

* Chiudi applicazioni in background durante i test.
* Mantieni la macchina “in idle” per risultati più stabili.
* Aumenta `repeats` (es. 10) se vuoi statistiche più affidabili (su Jetson, attenzione ai tempi).
* Per confronti, usa sempre **le stesse 3 immagini** e **stesse risoluzioni**.

---

## Troubleshooting

* **`ImportError: fastsam`** → `pip install "git+https://github.com/CASIA-IVA-Lab/FastSAM.git"`
* **CUDA non rilevata** → verifica driver (`nvidia-smi` su PC, `tegrastats` su Jetson); in fallback lo script userà CPU/MPS
* **OpenCV in conflitto** (Linux/Jetson) → `pip install opencv-python-headless`
* **Pesi non scaricati** → scarica manualmente `FastSAM-x.pt` / `FastSAM-s.pt` e mettili accanto allo script
* **RAM insufficiente** → usa solo `FastSAM-s.pt`, riduci risoluzioni e `repeats`

---

## FAQ

**Quante immagini servono?**
Almeno **3** in `images/`. Se ce ne sono di più, lo script prende le prime 3 ordinate alfabeticamente.

**Posso aggiungere altre risoluzioni?**
Sì, modificando la lista `sizes` nello script.

**Posso usare solo ORB o solo FastSAM?**
Sì: rimuovi i pesi per saltare FastSAM, oppure commenta la chiamata a ORB nello script.

**Dove trovo i pesi?**
Lo script tenta più URL/mirror. In alternativa, scaricali e mettili accanto allo script (stessi nomi file).

---

## Crediti

* **FastSAM** — CASIA-IVA-Lab (Segment Anything veloce per immagini 2D)
* **OpenCV** — libreria computer vision open-source

```

Se vuoi, posso anche **aggiornare lo script** per aggiungere una rilevazione automatica del Jetson e applicare la “Jetson Safe Mode” senza modifiche manuali al codice.
```
