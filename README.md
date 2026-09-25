<div align="center">

# 🏭 CNN Industrial Defect Detection System

### High-Performance Quality Control via Deep Learning

[![Language](https://img.shields.io/badge/Language-C%2B%2B17-blue?style=flat-square&logo=cplusplus)](https://en.cppreference.com/w/cpp/17)
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow?style=flat-square&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-LibTorch%20%7C%20TorchScript-orange?style=flat-square&logo=pytorch)](https://pytorch.org/cppdocs/)
[![CV Library](https://img.shields.io/badge/Vision-OpenCV%204.x-green?style=flat-square&logo=opencv)](https://opencv.org/)
[![Build System](https://img.shields.io/badge/Build-CMake%203.18%2B-red?style=flat-square&logo=cmake)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

*Automated surface defect detection for steel manufacturing — CNN inference in native C++ with a Python data preparation pipeline.*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation & Build](#installation--build)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Configuration Reference](#configuration-reference)
- [Model Checkpoints](#model-checkpoints)
- [Evaluation Metrics](#evaluation-metrics)
- [End-to-End Workflow Walkthrough](#end-to-end-workflow-walkthrough)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a complete, production-oriented pipeline for detecting surface defects on manufactured steel products. The system performs **binary classification** (`OK` / `DEFECT`) on grayscale steel-strip images drawn from the [Kaggle Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) dataset.

The **inference engine is written in C++17** using LibTorch (the C++ frontend of PyTorch), enabling near real-time performance suitable for integration into industrial inspection systems. A Python helper script handles raw Kaggle dataset organization before training.

---

## Key Features

| Feature | Detail |
|---------|--------|
| 🚀 **High-performance C++ inference** | Native LibTorch forward pass — no Python overhead at inference time |
| 🧠 **Custom CNN architecture** | 3-layer Conv+BN+ReLU extractor → Global Average Pool → FC classifier |
| 🗂️ **Modular pipeline** | Distinct modules for loading, preprocessing, inference, decision logic, visualization, and evaluation |
| 📊 **Rich evaluation metrics** | Accuracy, Precision, Recall (primary), F1, AUC, confusion matrix, ROC curve |
| 🎯 **Configurable decision threshold** | Decoupled `DecisionEngine` applies business-logic threshold independently of the model |
| 📽️ **Multiple input modes** | Single image, folder batch, webcam stream, built-in demo |
| 🔧 **YAML configuration** | All paths, hyperparameters, and thresholds defined in `config/config.yaml` |
| 🏷️ **TorchScript model format** | Models serialized as `.pt` TorchScript — portable and loadable without Python |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Industrial Defect Detection System               │
│                                                                      │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │ ImageLoader│───▶│ Preprocessor │───▶│   CNNModel   │             │
│  │            │    │              │    │ (TorchScript)│             │
│  │ • file     │    │ • resize     │    │              │             │
│  │ • folder   │    │ • BGR→RGB    │    │ • Conv×3     │             │
│  │ • webcam   │    │ • normalize  │    │ • BN, ReLU   │             │
│  └────────────┘    └──────────────┘    │ • GlobalPool │             │
│                                        │ • FC layers  │             │
│                                        └──────┬───────┘             │
│                                               │ logits / probs      │
│                                        ┌──────▼───────┐             │
│                                        │ DecisionEngine│             │
│                                        │ (threshold)   │             │
│                                        └──────┬───────┘             │
│                                               │ OK / DEFECT         │
│                              ┌────────────────┼────────────┐        │
│                              ▼                ▼            ▼        │
│                         ┌─────────┐    ┌──────────┐  ┌──────────┐  │
│                         │Visualizer│   │ Evaluator │  │  Logs /  │  │
│                         │ (OpenCV) │   │ (metrics) │  │  CSV out │  │
│                         └─────────┘    └──────────┘  └──────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

**Training Phase (C++ binary):**
```
DataLoader ──▶ Preprocessor ──▶ DefectNet (train) ──▶ LossFunctions ──▶ ModelTrainer
                                                                            │
                                                              ┌─────────────┘
                                                              ▼
                                                    checkpoints/ + models/defect_model.pt
```

---

## Repository Layout

```
CNN-industrial-defects-detection/
│
├── CMakeLists.txt              ← CMake build definition (C++17, LibTorch, OpenCV)
├── README.md                   ← This file
├── NOTES.md                    ← Developer notes, known issues, verification checklist
├── cahier-charge.md            ← Original project specification (French)
│
├── config/
│   ├── config.yaml             ← Runtime configuration (paths, thresholds, visualization)
│   └── training_config.yaml    ← Training hyperparameters (epochs, LR, optimizer, ...)
│
├── src/                        ← C++ implementation files
│   ├── main.cpp                ← Inference entry-point
│   ├── train_main.cpp          ← Training entry-point
│   ├── CNNModel.cpp            ← TorchScript model loading & forward pass
│   ├── DataLoader.cpp          ← LibTorch Dataset class, augmentation
│   ├── DecisionEngine.cpp      ← Business-logic threshold application
│   ├── Evaluator.cpp           ← Metrics: accuracy, precision, recall, F1, AUC
│   ├── ImageLoader.cpp         ← File / directory / webcam image acquisition
│   ├── LossFunctions.cpp       ← Cross-entropy, Focal loss, BCE-with-logits
│   ├── ModelTrainer.cpp        ← Training loop, LR scheduler, checkpointing
│   ├── Preprocessor.cpp        ← Resize, color convert, normalize, augment
│   └── Visualizer.cpp          ← OpenCV rendering, overlays, FPS, recording
│
├── include/                    ← C++ header files (one per module)
│   ├── CNNModel.h
│   ├── DataLoader.h
│   ├── DecisionEngine.h
│   ├── Evaluator.h
│   ├── ImageLoader.h
│   ├── LossFunctions.h
│   ├── ModelTrainer.h
│   ├── Preprocessor.h
│   └── Visualizer.h
│
├── data/
│   ├── organize_kaggle_data.py ← Python script to prepare the Kaggle dataset
│   ├── dataset_config.yaml     ← Auto-generated by organize script (dataset stats)
│   ├── dataset_summary.txt     ← Auto-generated human-readable summary
│   ├── train/                  ← Organized training images (gitignored)
│   ├── val/                    ← Organized validation images (gitignored)
│   └── test/                   ← Organized test images (gitignored)
│
├── models/
│   └── defect_model.pt         ← Final TorchScript model (gitignored)
│
├── checkpoints/
│   ├── best_model.pt           ← Best validation checkpoint (gitignored)
│   └── checkpoint_epoch_*.pt   ← Periodic epoch checkpoints (gitignored)
│
├── logs/                       ← Training logs (gitignored)
├── output/                     ← CSV training history, visualizations (gitignored)
└── results/                    ← Evaluation reports (gitignored)
```

---

## Prerequisites

### C++ Build Environment

| Requirement | Minimum Version | Notes |
|-------------|----------------|-------|
| **C++ Compiler** | C++17 (MSVC 2019, GCC 9, Clang 10) | Must support `std::filesystem` |
| **CMake** | 3.18+ | Required for modern target-based LibTorch detection |
| **LibTorch** | 2.0+ (CPU or CUDA) | Download from [pytorch.org](https://pytorch.org/get-started/locally/) → select **C++ / Java** |
| **OpenCV** | 4.5+ | Install via `vcpkg`, installer, or build from source |

### Python Environment (dataset preparation only)

| Package | Version | Purpose |
|---------|---------|---------|
| Python  | 3.9+    | Script interpreter |
| numpy   | ≥1.21   | Array shuffling for reproducible splits |
| pandas  | ≥1.3    | CSV parsing of Kaggle annotation file |
| pyyaml  | ≥5.4    | Writing `dataset_config.yaml` output |

```bash
pip install numpy pandas pyyaml
```

---

## Installation & Build

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CNN-industrial-defects-detection.git
cd CNN-industrial-defects-detection
```

### 2. Install LibTorch

Download the **Release** (CPU or CUDA) zip from [pytorch.org](https://pytorch.org/get-started/locally/)
and extract to `C:\libtorch\` (Windows) or `/opt/libtorch/` (Linux/macOS).

Resulting layout expected:
```
C:\libtorch\libtorch\
├── include\
├── lib\
└── share\cmake\Torch\TorchConfig.cmake   ← CMake looks for this
```

### 3. Configure with CMake

```powershell
# Windows (PowerShell) — adjust Torch_DIR if installed elsewhere
cmake -S . -B build `
    -DCMAKE_BUILD_TYPE=Release `
    -DTorch_DIR="C:/libtorch/libtorch/share/cmake/Torch"
```

```bash
# Linux / macOS
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DTorch_DIR="/opt/libtorch/share/cmake/Torch"
```

### 4. Build

```powershell
# Windows
cmake --build build --config Release --parallel

# Linux / macOS
cmake --build build --parallel $(nproc)
```

This produces two executables in `build/` (or `build/Release/` on MSVC):

| Binary | Purpose |
|--------|---------|
| `CNNIndustrialDefectsDetection` | Inference |
| `CNNIndustrialDefectsTraining`  | Training  |

---

## Dataset Preparation

The project targets the **Kaggle Severstal Steel Defect Detection** dataset.

### Step 1 — Download from Kaggle

```bash
# Requires Kaggle API token configured in ~/.kaggle/kaggle.json
pip install kaggle
kaggle competitions download -c severstal-steel-defect-detection -p data/
cd data && unzip severstal-steel-defect-detection.zip
```

Or download manually from [kaggle.com/c/severstal-steel-defect-detection](https://www.kaggle.com/c/severstal-steel-defect-detection)
and extract so you have:
```
data/
├── train_images/   ← ~12,568 JPEG images (1600×256 px each)
├── test_images/    ← unlabeled competition test images
└── train.csv       ← annotation file (ImageId, ClassId, EncodedPixels)
```

### Step 2 — Organize into train/val/test splits

```bash
# From the project root:
python data/organize_kaggle_data.py --seed 42

# Optional flags:
#   --data-dir PATH     Override default ./data input location
#   --output-dir PATH   Override default ./data/organized output location
#   --train-ratio 0.8   Training fraction (default: 0.8)
#   --val-ratio   0.1   Validation fraction (default: 0.1; remainder → test)
#   --yes               Skip confirmation prompt
```

Output:
```
data/organized/
├── train/
│   ├── OK/      ← images with no annotated defects
│   └── DEFECT/  ← images with at least one defect segment
├── val/
│   ├── OK/
│   └── DEFECT/
├── test/
│   ├── OK/
│   └── DEFECT/
├── dataset_summary.txt
└── dataset_config.yaml
```

### Step 3 — Verify paths in config

Open `config/config.yaml` and confirm the paths section matches your output:
```yaml
paths:
  train_img: "data/organized/train/"
  val_img:   "data/organized/val/"
  test_img:  "data/organized/test/"
```

---

## Training

```powershell
# From project root (make sure current directory is correct — paths are relative)
cd build/Release       # Windows MSVC
.\CNNIndustrialDefectsTraining.exe --config config/training_config.yaml

# Or from project root on Linux:
./build/CNNIndustrialDefectsTraining --config config/training_config.yaml
```

**What happens:**
1. Validates `data/organized/train/` exists and is non-empty.
2. Creates output directories (`models/`, `checkpoints/`, `logs/`, `output/`).
3. Initializes a 3-layer CNN with BatchNorm + Global Average Pooling.
4. Trains for `numEpochs` epochs using Adam optimizer + StepLR scheduler.
5. Saves the best validation-accuracy model to `checkpoints/best_model.pt`.
6. Saves the final model to `models/defect_model.pt` (TorchScript format).
7. Exports per-epoch metrics to `output/training_history.csv`.

> **GPU Note:** Set `useGPU: true` in `config/training_config.yaml` if a
> CUDA-enabled LibTorch build is installed. Training will fall back to CPU
> automatically if CUDA is unavailable.

---

## Inference

### Demo mode (no model required)

```bash
./CNNIndustrialDefectsDetection --mode demo
```

Tests the preprocessing pipeline on a synthetic image and displays the
before/after side-by-side in an OpenCV window.

### Single image

```bash
./CNNIndustrialDefectsDetection \
    --mode  image \
    --input data/organized/test/DEFECT/0002cc93b.jpg
```

### Folder / batch evaluation

```bash
./CNNIndustrialDefectsDetection \
    --mode  folder \
    --input data/organized/test/
```

Ground-truth labels are inferred from the parent folder name (`OK` or `DEFECT`).
Final Accuracy, Precision, Recall, F1, and a confusion matrix are printed on exit.

### With a custom config

```bash
./CNNIndustrialDefectsDetection \
    --config config/config.yaml \
    --mode   folder \
    --input  data/organized/test/
```

---

## Configuration Reference

### `config/config.yaml` — Runtime settings

| Key | Default | Description |
|-----|---------|-------------|
| `paths.train_img` | `data/organized/train/` | Training image root |
| `paths.model_save_path` | `models/best_model.pt` | TorchScript model path |
| `img_config.input_size` | `[224, 224]` | Resize target (W × H) |
| `img_config.normalization.mean` | ImageNet mean | Per-channel normalization mean |
| `img_config.normalization.std` | ImageNet std | Per-channel normalization std |
| `thresholds.defect_threshold` | `0.75` | Min defect probability to classify as DEFECT |
| `thresholds.uncertainty_margin` | `0.10` | Window around threshold → UNCERTAIN label |
| `runtime.mode` | `image` | Default mode: `demo`, `image`, `folder`, `webcam` |
| `runtime.device` | `cpu` | `cpu` or `cuda` |

### `config/training_config.yaml` — Training settings

| Key | Default | Description |
|-----|---------|-------------|
| `numEpochs` | `50` | Training epochs |
| `batchSize` | `32` | Mini-batch size |
| `learningRate` | `0.001` | Initial learning rate |
| `optimizer` | `ADAM` | `SGD`, `ADAM`, `RMSPROP`, `ADAGRAD` |
| `lossFunction` | `CROSS_ENTROPY` | `CROSS_ENTROPY`, `FOCAL_LOSS`, `BCE_WITH_LOGITS` |
| `lrScheduler` | `STEP_LR` | `STEP_LR`, `EXPONENTIAL_LR`, `COSINE_ANNEALING_LR` |
| `useGPU` | `false` | Enable CUDA acceleration |
| `useEarlyStopping` | `true` | Stop if val loss stagnates |
| `earlyStoppingPatience` | `10` | Epochs before early stop triggers |

---

## Model Checkpoints

| File | Description |
|------|-------------|
| `checkpoints/best_model.pt` | Saved at the epoch with highest validation accuracy |
| `checkpoints/checkpoint_epoch_N.pt` | Periodic checkpoint every `checkpointInterval` epochs |
| `models/defect_model.pt` | Final model after all epochs — used by the inference binary |

> **Important:** These files are in **TorchScript** (`.pt`) format, not PyTorch
> Python pickle format (`.pth`). If you have a `.pth` checkpoint from Python
> training, you must convert it:
> ```python
> model = YourModel(); model.load_state_dict(torch.load("best.pth"))
> model.eval(); torch.jit.script(model).save("models/defect_model.pt")
> ```

---

## Evaluation Metrics

The `Evaluator` module computes all standard binary classification metrics,
with emphasis on **Recall** (critical in manufacturing: missing a defect is
far worse than a false alarm):

| Metric | Formula | Importance |
|--------|---------|-----------|
| **Recall** (Sensitivity) | TP / (TP + FN) | 🔴 Primary — defect escape rate |
| **Precision** | TP / (TP + FP) | How reliable are defect alarms? |
| **F1 Score** | 2 × Prec × Recall / (Prec + Recall) | Balanced summary |
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Specificity** | TN / (TN + FP) | OK identification rate |
| **AUC** | Area under ROC | Threshold-independent performance |

Results are saved to `results/` as CSV and printed to stdout.

---

## End-to-End Workflow Walkthrough

> This section describes the full expected execution flow.
> **No commands were run during code review** — see [NOTES.md](NOTES.md) for
> manual verification items before attempting this workflow.

### Phase 1 — Build

```powershell
# 1. Clone repo
git clone https://github.com/<user>/CNN-industrial-defects-detection.git
cd CNN-industrial-defects-detection

# 2. Configure CMake (adjust Torch_DIR to your LibTorch install)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTorch_DIR="C:/libtorch/libtorch/share/cmake/Torch"

# 3. Compile (parallel)
cmake --build build --config Release --parallel

# Expected output:
#   [1/14] Building CXX object CMakeFiles/...
#   ...
#   [14/14] Linking CXX executable CNNIndustrialDefectsDetection.exe
```

**Likely compiler warnings to watch for:**
- `warning C4244` (MSVC): narrowing conversion in OpenCV/LibTorch — benign
- `warning: unused parameter` — benign in stub methods
- `error: 'torch::jit::script::Module'` undefined: resolved by `#include <torch/script.h>` in `CNNModel.h` ✓ (already fixed)

### Phase 2 — Data Preparation

```bash
# Download Kaggle dataset to data/
python data/organize_kaggle_data.py --seed 42

# Verify output:
ls data/organized/train/OK/        # Should contain images
ls data/organized/train/DEFECT/    # Should contain images
cat data/organized/dataset_summary.txt
```

**Potential issues:**
- `train.csv` with all-defect rows → `OK/` folder will be empty → model will predict all DEFECT
- Large dataset (~12k images, ~3.4 GB) — ensure sufficient disk space in `data/organized/`

### Phase 3 — Train

```powershell
cd build/Release
.\CNNIndustrialDefectsTraining.exe --config ../../config/training_config.yaml
```

**Expected console output:**
```
╔══════════════════════════════════════════════════════════╗
║   Industrial Defect Detection System - Training Module   ║
...
🔍 Validating environment...
✅ CUDA available: GPU acceleration enabled   (or: ⚠️ CUDA not available)
📊 Setting up DataLoader...
✅ DataLoader ready
   Training samples: 8532
   Validation samples: 1067
🧠 Creating CNN model...
Model Architecture:
  Input: 3x224x224 (RGB image)
  ...
🚀 Starting training...
Epoch [1/50]: Loss=0.693 Acc=52.1% | Val Loss=0.681 Val Acc=56.4%
...
✅ Training completed successfully!
   Model saved: models/defect_model.pt
```

### Phase 4 — Inference

```powershell
# Demo (no model needed)
.\CNNIndustrialDefectsDetection.exe --mode demo

# Batch evaluation with metrics
.\CNNIndustrialDefectsDetection.exe --mode folder --input ../../data/organized/test/
```

---

## Contributing

1. Fork the repository and create a feature branch: `git checkout -b feat/yaml-config-parser`
2. Make your changes following the existing module structure.
3. Ensure all new C++ code compiles at `-Wall -Wextra` without warnings.
4. Add or update inline Doxygen comments for any new public API.
5. Open a Pull Request describing the change and its motivation.

**Areas open for contribution:**
- YAML config parser integration (yaml-cpp)
- Webcam streaming mode in `ImageLoader.cpp`
- Grad-CAM heatmap visualization in `Visualizer.cpp`
- Cross-platform `ctime_s` fix in `Visualizer.h`
- REST API wrapper for edge deployment

---

## License

This project is released under the **MIT License**.  
See [LICENSE](LICENSE) for details.

---

<div align="center">

*Built as part of an industrial computer vision curriculum project.*  
*Dataset: [Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) — Kaggle*

</div>
