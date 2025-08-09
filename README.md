
# Benchmark FastSAM + ORB (Multipiattaforma)

Script di benchmark per:
- **FastSAM-x** e **FastSAM-s** a risoluzioni: `512, 576, 640, 704, 768, 832, 896, 960, 1024` su **8 immagini statiche**.
- **ORB (OpenCV)** con sweep di combinazioni parametriche, misurando **latenza**, **#keypoint** e **keypoints/ms**.
- **ORB per Pose estimation e Orientamento** stima orientamento e angolo di immagini ruotate di 45 gradi (front/back) + angolo usando due reference:

	•	Coop_fronte_texture_fronte.png (**front**)

 	•	Coop_retro_texture_retro.png (**back**)
<img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/2836cc11-2758-48c8-8f54-1132330a1dc9" />







**Output:**  CSV con i tempi e risultati.

---

## Indice
- [Requisiti](#requisiti)
- [Piattaforme supportate](#piattaforme-supportate)
- [Setup ambiente](#setup-ambiente)
- [Installazione PyTorch](#installazione-pytorch)
- [Installazione dipendenze](#installazione-dipendenze)
- [Struttura progetto](#struttura-progetto)
- [Uso](#uso)
- [Uso su Jetson (Nano/Orin/Xavier)](#uso-su-jetson-nanoorinxaiver)
- [Ottimizzazioni & consigli](#ottimizzazioni--consigli)
- [Troubleshooting](#troubleshooting)

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

---

## Uso

Esegui senza argomenti:

```bash
python benchmark_fastsam_orb.py
```

* Device scelto automaticamente (CUDA/MPS/CPU).
* Se i pesi non sono presenti, lo script prova a **scaricarli** (vedi sotto).


## Uso su Jetson (Nano/Orin/Xavier)

I dispositivi Jetson hanno RAM/CPU limitate; ecco il percorso consigliato.

### 1) Ambiente & dipendenze

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel
pip install opencv-python-headless numpy pandas pillow tqdm
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
