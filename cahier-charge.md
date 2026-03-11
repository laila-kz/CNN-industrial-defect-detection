# Project Specification (Cahier des Charges)

## Project Title

Industrial Defect Detection System using Convolutional Neural Networks (CNN)

---

## 1. Project Context and Motivation

In modern manufacturing environments, quality control is a critical step. Manual inspection of products is time-consuming, costly, and prone to human error, especially when defects are small or when inspection must be done continuously.

The goal of this project is to design and implement an **automated vision-based defect detection system** capable of identifying surface defects on manufactured products using **Convolutional Neural Networks (CNNs)**. The system is intended to simulate a real industrial scenario where images are captured from a camera and analyzed in near real-time.

---

## 2. Project Objectives

### 2.1 Main Objective

Develop a CNN-based computer vision system that can automatically classify products as **OK** or **DEFECTIVE** based on input images.

### 2.2 Secondary Objectives

* Apply image preprocessing techniques to improve model robustness
* Implement a CNN model using a non-Python language (C++ or Java)
* Perform real-time or near real-time inference
* Provide visual feedback and performance metrics
* Evaluate the system using appropriate classification metrics

---

## 3. Scope of the Project

### Included in the scope

* Image-based defect detection
* Binary classification (OK / DEFECT)
* Offline training and online inference
* Visualization of predictions
* Evaluation of model performance

### Excluded from the scope

* Physical camera calibration
* Hardware-level integration with industrial machines
* Large-scale cloud deployment

---

## 4. Functional Requirements

### FR1 – Image Acquisition

The system shall be able to:

* Load images from a directory (training/testing)
* Capture frames from a video file or webcam

### FR2 – Image Preprocessing

The system shall apply the following preprocessing steps:

* Resize images to a fixed resolution
* Convert color format (BGR to RGB if necessary)
* Normalize pixel values
* Optionally apply noise reduction or contrast enhancement

### FR3 – CNN-Based Classification

The system shall:

* Use a CNN to extract visual features
* Output class probabilities (OK / DEFECT)
* Support a configurable decision threshold

### FR4 – Decision Logic

The system shall:

* Classify products based on CNN output probabilities
* Separate prediction (model) from decision rules (business logic)

### FR5 – Visualization

The system shall display:

* Input image or video frame
* Predicted label (OK / DEFECT)
* Confidence score
* Color-coded result
* Processing speed (FPS)

### FR6 – Evaluation

The system shall compute:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

---

## 5. Non-Functional Requirements

### NFR1 – Performance

* The system should be capable of near real-time inference
* Inference time should be minimized

### NFR2 – Modularity

* The system shall be modular (separation of preprocessing, model, logic, and visualization)

### NFR3 – Maintainability

* The codebase shall be well-structured and documented

### NFR4 – Portability

* The system shall be buildable on a standard desktop environment

---

## 6. Technical Constraints

* Programming Language: **C++ (preferred) or Java**
* Deep Learning Framework:

  * C++: LibTorch
  * Java: Deeplearning4j
* Computer Vision Library: OpenCV
* Dataset: Public industrial defect datasets

---

## 7. Dataset Description

The system will use publicly available datasets containing labeled images of industrial surface defects.

Examples:

* NEU Surface Defect Dataset
* MVTec Anomaly Detection Dataset (subset)

Each image is labeled as:

* OK (no defect)
* DEFECT (one or more defects present)

---

## 8. System Architecture

High-level architecture:

Image Source → Preprocessing → CNN Model → Decision Logic → Visualization / Output

The architecture must clearly separate training and inference phases.

---

## 9. Training and Inference Workflow

### Training Phase

* Dataset loading and preprocessing
* CNN training using labeled data
* Model evaluation and saving

### Inference Phase

* Load trained model
* Process incoming images or frames
* Perform forward pass only (no training)
* Apply decision logic and display results

---

## 10. Evaluation Criteria

The project will be evaluated based on:

* Correctness of implementation
* Quality of preprocessing
* Model performance (especially recall)
* Code organization and clarity
* Quality of documentation and explanation

---

## 11. Deliverables

* Source code of the project
* Trained CNN model file
* Project report / README
* Visual results (screenshots or demo video)

---

## 12. Expected Outcomes

At the end of the project, the system should be able to:

* Automatically detect surface defects from images
* Operate in a realistic industrial inspection scenario
* Demonstrate both deep learning knowledge and software engineering skills

---

## 13. Future Extensions (Optional)

* Multi-class defect classification
* Grad-CAM or explainability visualization
* Multithreaded inference
* REST API integration
* Transfer learning with pretrained models
