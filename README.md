<div align="center">

# Self-Driving Car Using AI

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](#)
[![Webots](https://img.shields.io/badge/Webots-Simulator-00A6FF?style=flat-square)](#)

**End-to-end self-driving pipeline — from computer vision fundamentals to behavioral cloning — using Webots simulation, CNNs, and real-time LiDAR-based collision avoidance.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Modules](#modules)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Module Details](#module-details)
  - [Module 0 — Installation](#module-0--installation)
  - [Module 1 — Python Fundamentals](#module-1--python-fundamentals)
  - [Module 2 — Computer Vision](#module-2--computer-vision)
  - [Module 3 — Road Sign Classification](#module-3--road-sign-classification)
  - [Module 4 — Collision Avoidance](#module-4--collision-avoidance)
  - [Module 5 — Behavioral Cloning](#module-5--behavioral-cloning)
- [Project Structure](#project-structure)

---

## Overview

This project implements a progressive, modular self-driving car pipeline in the Webots 3D simulation environment. Each module builds on the previous one, advancing from basic image processing to a fully autonomous CNN-based driving system.

The pipeline covers:
1. **Perception** — Camera and LiDAR sensor processing
2. **Classification** — Road sign recognition using HOG+SVM and CNN
3. **Control** — PID-based lane following and reactive obstacle avoidance
4. **End-to-End Driving** — Behavioral cloning with deep learning

---

## Project Architecture

```
┌───────────────────────────────────────────────────────┐
│                  Webots Simulator                      │
│        City environment with vehicle + sensors         │
├───────────┬──────────────┬────────────────────────────┤
│  Camera   │    LiDAR     │     Vehicle API             │
│  (RGB)    │  (Sick LMS)  │  (Steering + Throttle)     │
├───────────┴──────────────┴────────────────────────────┤
│                                                        │
│  Module 2: Image Thresholding → PID Lane Following     │
│  Module 3: HOG Features → SVM/CNN Road Sign Classifier │
│  Module 4: LiDAR Point Cloud → Reactive Avoidance      │
│  Module 5: Camera → CNN → Steering Angle Prediction    │
│                                                        │
└───────────────────────────────────────────────────────┘
```

---

## Modules

| Module | Topic | Key Techniques |
|--------|-------|---------------|
| **0** | Installation & Setup | Environment, dependencies, Webots |
| **1** | Python Fundamentals | NumPy, Matplotlib, OpenCV basics |
| **2** | Computer Vision | Image thresholding, PID camera control |
| **3** | Road Sign Classification | HOG features, SVM, CNN (TensorFlow) |
| **4** | Collision Avoidance | 2D LiDAR processing, reactive control |
| **5** | Behavioral Cloning | Dataset collection, CNN training, autonomous driving |

---

## Technology Stack

| Category | Tools |
|----------|-------|
| Simulation | Webots (3D robotics simulator) |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV, scikit-image |
| Machine Learning | scikit-learn (SVM, preprocessing) |
| Data Processing | NumPy, Pandas, Matplotlib |
| Input Handling | Xbox controller / Keyboard (via `inputs` / `pynput`) |

---

## Prerequisites

- Python 3.8+
- [Webots](https://cyberbotics.com/) simulator (R2023+)
- pip package manager
- Optional: Xbox controller for manual data collection

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/nuhel7050/Self-Driving-Car-Using-AI.git
cd Self-Driving-Car-Using-AI
```

### 2. Set Up Python Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r 0_install/requirements.txt
```

### 3. Configure Webots

1. Open Webots and load a world file from `worlds/`:
   - `city.wbt` — Open city environment
   - `city_with_borders.wbt` — City with lane borders
2. Set the controller for the vehicle to the desired Python script.

---

## Usage

### Running a Simulation Controller

1. Open Webots → Load world from `worlds/`
2. Set the vehicle's controller to the target Python script
3. Run the simulation

### Training a Self-Driving Model (Module 5)

```bash
# Step 1: Collect training data (drive manually via controller)
cd 5_behavioural_coloning
python 1_create_dataset.py

# Step 2: Analyze the dataset
jupyter notebook 2_dataset_insights.ipynb

# Step 3: Train the CNN model
python 3_train.py

# Step 4: Run the autonomous driver
python 4_run_model.py
```

---

## Module Details

### Module 0 — Installation

**Path:** `0_install/`

Environment setup scripts and dependency management.

| File | Purpose |
|------|---------|
| `requirements.txt` | Python package dependencies |
| `install_linux.sh` | Automated Linux setup script |
| `general.yml` | Conda environment specification |

---

### Module 1 — Python Fundamentals

**Path:** `1_python/`

Jupyter notebooks covering foundational libraries used throughout the project.

| Notebook | Topic |
|----------|-------|
| `1_np.ipynb` | NumPy array operations and math |
| `2_matplotlib.ipynb` | Data visualization |
| `3_opencv.ipynb` | Image loading, manipulation, drawing |

---

### Module 2 — Computer Vision

**Path:** `2_computer_vision/`

Camera-based perception and control in Webots simulation.

| File | Description |
|------|-------------|
| `1_image_threshold.py` | Binary image thresholding for lane detection |
| `2_camera_pid.py` | PID controller for camera-based lane following |
| `utils.py` | Image acquisition and preprocessing utilities |

**Pipeline:**
```
Camera Image → Grayscale → Threshold → Centroid Detection → PID → Steering
```

---

### Module 3 — Road Sign Classification

**Path:** `3_road_sign_classification/`

Three approaches to traffic sign classification, progressively increasing in sophistication.

| File | Approach | Description |
|------|----------|-------------|
| `1_hog.ipynb` | HOG Features | Feature extraction using Histogram of Oriented Gradients |
| `2_hog_svm.py` | HOG + SVM | Classical ML pipeline: HOG features → SVM classifier |
| `3_cnn.ipynb` | CNN | Deep learning classifier using TensorFlow/Keras |
| `preprocessing.py` | Data Pipeline | Dataset loading, augmentation, and preprocessing |

---

### Module 4 — Collision Avoidance

**Path:** `4_collision_avoidance/`

LiDAR-based obstacle detection and avoidance using the Sick LMS 291 2D laser scanner.

| File | Description |
|------|-------------|
| `1_lidar_manual.py` | Manual driving with LiDAR visualization (bird's-eye view) |
| `2_lidar_automated.py` | Automated reactive collision avoidance |
| `driving_inputs.py` | Xbox/keyboard controller abstraction |

**LiDAR Processing Pipeline:**
```
Raw Range Image → Polar-to-Cartesian → Bird's Eye View → Obstacle Zones → Steering Decision
```

---

### Module 5 — Behavioral Cloning

**Path:** `5_behavioural_coloning/`

End-to-end learning: a CNN learns to predict steering angles directly from camera images, trained on human driving demonstrations.

| File | Stage | Description |
|------|-------|-------------|
| `1_create_dataset.py` | Data Collection | Record camera images + steering angles while driving manually |
| `2_dataset_insights.ipynb` | Analysis | Visualize dataset distribution and balance |
| `3_train.py` | Training | Train CNN with classification (left/straight/right) |
| `3_train_reg.py` | Training (Alt) | Train CNN with regression (continuous angle) |
| `4_run_model.py` | Inference | Autonomous driving using the trained model |
| `5_explain.ipynb` | Explainability | Model interpretation and visualization |
| `utils.py` | Utilities | Image preprocessing, model loading, profiling |
| `driving_inputs.py` | Input | Xbox/keyboard controller abstraction |

**CNN Architecture (Classification):**
```
Input (64×128×3)
  → Conv2D(64, 7×7) → MaxPool
  → Conv2D(128, 3×3) → Conv2D(128, 3×3) → MaxPool
  → Conv2D(256, 3×3) → Conv2D(256, 3×3) → MaxPool
  → Flatten
  → Dense(128) → Dropout(0.2)
  → Dense(64) → Dropout(0.2)
  → Dense(3, softmax)  →  [Left, Straight, Right]
```

**Training:** SGD optimizer, categorical cross-entropy loss, early stopping (patience=3), TensorBoard logging.

---

## Project Structure

```
Self-Driving-Car-Using-AI/
├── 0_install/
│   ├── requirements.txt
│   ├── install_linux.sh
│   └── general.yml
│
├── 1_python/
│   ├── 1_np.ipynb
│   ├── 2_matplotlib.ipynb
│   └── 3_opencv.ipynb
│
├── 2_computer_vision/
│   ├── 1_image_threshold.py
│   ├── 2_camera_pid.py
│   └── utils.py
│
├── 3_road_sign_classification/
│   ├── 1_hog.ipynb
│   ├── 2_hog_svm.py
│   ├── 3_cnn.ipynb
│   └── preprocessing.py
│
├── 4_collision_avoidance/
│   ├── 1_lidar_manual.py
│   ├── 2_lidar_automated.py
│   └── driving_inputs.py
│
├── 5_behavioural_coloning/
│   ├── 1_create_dataset.py
│   ├── 2_dataset_insights.ipynb
│   ├── 3_train.py
│   ├── 3_train_reg.py
│   ├── 4_run_model.py
│   ├── 5_explain.ipynb
│   ├── utils.py
│   └── driving_inputs.py
│
├── worlds/
│   ├── city.wbt
│   └── city_with_borders.wbt
│
└── LICENSE
```
