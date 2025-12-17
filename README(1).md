# ML Classifier for Fault Diagnosis in Rotary Machines

This project implements a machine learning pipeline for **fault diagnosis in rotary machines** using vibration data.  
It supports **fault detection** and **fault classification** across multiple operating speeds using both **time-domain** and **frequency-domain (FFT)** feature extraction.

The system is designed to be modular and interactive, allowing the user to select:
- Operating speed (RPM)
- Feature extraction method
- Machine learning model
- Diagnosis task

---

## Project Overview

**Input**
- Vibration signal data stored as CSV files
- Data collected at different rotational speeds (25, 50, 75 RPM)

**Processing**
- Time-domain feature extraction
- Frequency-domain feature extraction using FFT

**Models**
- Decision Tree
- Support Vector Machine (SVM)
- Logistic Regression
- Linear Regression
- Neural Network

**Tasks**
- **Fault Detection** (Fault vs No Fault)
- **Fault Classification** (39 fault categories)

**Output**
- Classification metrics (accuracy, precision, recall, F1-score)
- Confusion matrices
- Optional saved result logs

---

## Requirements

- Python 3.x  
- Packages:
  - pandas
  - numpy
  - scipy
  - scikit-learn

All dependencies can be installed automatically using `requirements.txt`.

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/USERNAME/REPOSITORY_NAME.git
cd REPOSITORY_NAME
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the project
```bash
python main.py
```
