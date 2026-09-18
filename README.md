# Deep-Learning
# 🦜 BirdCLEF+ 2026 — Bioacoustic Species Recognition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=for-the-badge&logo=pytorch)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx)
![Kaggle](https://img.shields.io/badge/Kaggle-Score%200.946-20BEFF?style=for-the-badge&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Kaggle Competition | Cornell Lab of Ornithology & Google | 2025–2026**

[📄 Documentation](#-documentation) • [🚀 Quick Start](#-quick-start) • [🏗️ Architecture](#️-architecture) • [📊 Results](#-results) • [👥 Team](#-team)

</div>

---

## 📌 Overview

This repository contains our solution for the **BirdCLEF+ 2026 Kaggle Competition**, focused on automated identification of wildlife species from passive acoustic monitoring (PAM) recordings.

> **Task:** Given 60-second soundscape audio files, predict the presence or absence of **200+ species** (birds, amphibians, insects, mammals) for each 5-second window.

> **Result: 🏆 Public Leaderboard Score — 0.946 Macro ROC-AUC**

---

## 👥 Team

| Name | Roll No | Department |
|------|---------|-----------|
| Sheikh Abrar | 2022-SE-21 | Software Engineering |
| Nasir Abbas | 2022-SE-37 | Software Engineering |
| Mudassar Raza | 2022-SE-40 | Software Engineering |

**University:** The University of Azad Jammu and Kashmir, Muzaffarabad
**Course:** Deep Learning (SE-3203) | Semester 6th | Session 2023–2027
**Submitted To:** Engr. Ahmed Khawaja

---

## 📁 Project Structure

```
birdclef-2026/
│
├── 📓 notebooks/
│   └── birdclef2026_optimized.ipynb     # Main Kaggle notebook
│
├── 📂 src/
│   ├── models/
│   │   ├── proto_ssm.py                 # ProtoSSM architecture
│   │   ├── residual_ssm.py              # ResidualSSM error correction
│   │   └── selective_ssm.py             # Selective SSM (Mamba-inspired)
│   │
│   ├── features/
│   │   ├── perch_inference.py           # Google Perch v2 backbone
│   │   ├── mlp_probes.py                # Per-class MLP probes
│   │   └── prior.py                     # Bayesian ecological prior
│   │
│   ├── training/
│   │   ├── train_proto.py               # ProtoSSM training loop
│   │   └── train_residual.py            # ResidualSSM training
│   │
│   └── inference/
│       ├── tta.py                       # Test time augmentation
│       ├── sed_inference.py             # SED model inference
│       ├── blend.py                     # Rank-percentile blending
│       └── postprocess.py              # Post-processing gates
│
├── 📂 docs/
│   └── BirdCLEF2026_Report.pdf          # Full project documentation
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/birdclef-2026.git
cd birdclef-2026
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Required Data

```bash
# Kaggle competition data
kaggle competitions download -c birdclef-2026

# Required datasets
kaggle datasets download tuckerarrants/perch-v2-no-dft-onnx
kaggle datasets download tuckerarrants/bc2026-distilled-sed-public
kaggle datasets download jaejohn/perch-meta
```

### 4. Run the Notebook

Open `notebooks/birdclef2026_optimized.ipynb` on Kaggle or locally and run all cells.

---

## 🏗️ Architecture

### Full Pipeline

```
Raw Audio (.ogg, 60s)
        │
        ▼
┌─────────────────────┐
│  Google Perch v2    │  ← Frozen backbone (ONNX, 150x faster)
│  (1536-dim embeds)  │
└─────────────────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
Embeddings  Logits (14,795 species)
   │         │
   └────┬────┘
        │
        ▼
┌─────────────────────┐
│  Bayesian Prior     │  ← Site × Hour ecological statistics
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  MLP Probes         │  ← Per-class sklearn MLP on PCA embeddings
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ProtoSSM           │  ← BiSSM + Cross-Attention + Prototypes
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ResidualSSM        │  ← Second-pass error correction
└─────────────────────┘
        │
   ┌────┴────────────────┐
   │                     │
   ▼                     ▼
ProtoSSM (55%)      SED Model (45%)
                    5-fold EfficientNet
   │                     │
   └────────┬────────────┘
            │
            ▼
┌─────────────────────┐
│ Rank-Percentile     │  ← Removes score distribution mismatch
│ Blend               │
└─────────────────────┘
            │
            ▼
┌─────────────────────┐
│ Post-Processing     │  ← 5 Gates (noise, temporal, SED spike,
│ Gates (1–5)         │    sonotype mirror, rare-class threshold)
└─────────────────────┘
            │
            ▼
    submission.csv
```

---

## 🔬 Key Components

### 1. 🎙️ Google Perch v2 Backbone
- Frozen pretrained model — no fine-tuning
- Outputs **1536-dim embeddings** + 14,795 species logits
- ONNX no-DFT variant used for **150x CPU speedup**
- Species mapped via scientific name → Perch index

### 2. 🧠 ProtoSSM Model
- **Bidirectional Selective SSM** (Mamba-inspired) for temporal modeling
- **Multi-Head Cross-Attention** for window interactions
- **Prototypical learning** — one prototype per species class
- **Site + Hour embeddings** for ecological context
- **Multi-seed ensemble** (3 seeds) + **5-shift TTA**

### 3. 📡 Sound Event Detection (SED)
- EfficientNet-based CNN (distilled, 5-fold ensemble)
- Input: Log-mel spectrogram (256 mels)
- 50% clip-level + 50% frame-level fusion
- Gaussian temporal smoothing (σ=0.65)

### 4. 📊 Bayesian Ecological Prior
- 3-tier shrinkage: Global → Site → Hour → Site×Hour
- Applied as logit offset before sigmoid (λ=0.4)
- Encodes species geographic and temporal distribution

### 5. 🔀 Rank-Percentile Blending
```python
rank_proto = percentile_rank(proto_scores)
rank_sed   = percentile_rank(sed_scores)
final      = 0.55 * rank_proto + 0.45 * rank_sed
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Public Leaderboard** | **0.946 Macro ROC-AUC** |
| Evaluation Metric | Macro-averaged ROC-AUC |
| Species Count | 200+ |
| Hardware | CPU-only (Kaggle, 9h limit) |

---

## 📦 Requirements

```txt
torch>=2.0.0
tensorflow==2.20.0
onnxruntime>=1.16.0
librosa>=0.10.0
soundfile>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
scipy>=1.10.0
```

---

## 🗂️ External Resources Used

| Resource | Purpose |
|----------|---------|
| `google/bird-vocalization-classifier` | Perch v2 backbone + species labels |
| `tuckerarrants/perch-v2-no-dft-onnx` | Fast ONNX Perch inference |
| `tuckerarrants/bc2026-distilled-sed-public` | 5-fold EfficientNet SED models |
| `jaejohn/perch-meta` | Pre-computed Perch embeddings cache |
| `rishikeshjani/perch-onnx-for-birdclef-2026` | Custom ONNX Runtime wheel |

---

## 📄 Documentation

Full project documentation available in [`docs/BirdCLEF2026_Report.pdf`](docs/BirdCLEF2026_Report.pdf)

Includes:
- Complete architecture diagrams
- Training strategy details
- Feature engineering explanation
- Results and key learnings

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**BirdCLEF+ 2026 | University of Azad Jammu & Kashmir | Deep Learning Project**

Made with ❤️ by Sheikh Abrar, Nasir Abbas & Mudassar Raza

</div>
