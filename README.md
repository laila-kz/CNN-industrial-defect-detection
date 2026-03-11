# CNN Industrial Defect Detection

## Project Overview

This project implements a **Convolutional Neural Network (CNN)** to automatically detect and classify **industrial surface defects**.
The system is trained on an industrial inspection dataset and aims to assist quality control processes by identifying defective materials.

The project includes:

* Dataset preprocessing and organization
* Model training using deep learning
* Performance tracking and evaluation
* Structured project architecture for reproducibility

Such systems are commonly used in **automated manufacturing inspection pipelines** to improve product quality and reduce manual inspection effort.

---

## Project Structure

```
CNN-industrial-defects-detection
│
├── config/              # Configuration files
├── data/                # Dataset structure (dataset not included)
│   ├── train/
│   ├── val/
│   └── test/
│
├── include/             # Header files (C++ components if used)
├── src/                 # Source code
├── models/              # Trained model weights (not included)
├── output/              # Training outputs
├── logs/                # Training logs
│
├── organize_kaggle_data.py
├── dataset_config.yaml
├── CMakeLists.txt
└── README.md
```

---

## Dataset

The dataset used in this project comes from Kaggle.

Dataset link:
**[[Insert Google Drive Link Here]](https://www.kaggle.com/c/severstal-steel-defect-detection)**


Because of size limitations, the dataset is **not included in this repository**.

After downloading the dataset, organize it using:

```
python organize_kaggle_data.py
```

Expected dataset structure:

```
data/
   train/
   val/
   test/
```

---

## Trained Model

The trained CNN model weights are available here:

**Google Drive link:**
**[[Insert Google Drive Link Here]](https://drive.google.com/drive/folders/1984VXRz_nx7pguJJKlnbkhw0rgyR4NCb)**

Download the model and place it inside:

```
models/
```

Example:

```
models/defect_model.pt
```

---

## Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/CNN-industrial-defects-detection.git
cd CNN-industrial-defects-detection
```

Install dependencies (example):

```
pip install torch torchvision numpy pandas matplotlib
```

---

## Training the Model

To train the CNN model:

```
python train.py
```

Training logs and metrics will be saved in:

```
logs/
output/
```

---

## Model Architecture

The project uses a **Convolutional Neural Network (CNN)** architecture designed for image classification.

Typical CNN components used include:

* Convolutional layers
* ReLU activation
* Pooling layers
* Fully connected layers
* Softmax output for defect classification

---

## Applications

Industrial defect detection models like this can be applied to:

* Steel surface inspection
* Manufacturing quality control
* Automated visual inspection
* Smart factories and Industry 4.0 systems

---

## Author
Leila Khezaz
Developed as a deep learning project for industrial defect classification.

---


This project is for **educational and research purposes**.
