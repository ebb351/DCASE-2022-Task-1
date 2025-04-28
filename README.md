# Machine Listening Final Project: DCASE 2022 Task 1

## Overview
This project implements a baseline model for DCASE 2022 Task 1: Acoustic Scene Classification with Multiple Devices. The goal is to classify acoustic scenes using audio recordings from multiple devices, addressing the challenge of device mismatch in acoustic scene classification.

## Project Structure
```
.
├── data/               # Directory for storing datasets (git-ignored)
├── models/            # Model implementations and saved models
├── preprocessing/     # Data preprocessing scripts
└── outputs/          # Training outputs, logs, and results
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation
1. Clone this repository:
```bash
git clone https://github.com/ebb351/DCASE-2022-Task-1/tree/main/models
cd DCASE-2022-Task-1
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Sources

### Raw Waveform Data
[Add description and instructions for raw waveform data]

### Mel Spectrograms
Download from [Google Drive](https://drive.google.com/drive/folders/1m4in9I8e7DtPnJBLo7CYKqONKVYUwmr6)

Relevant Files:
- Training data: `DCASE2022_train.npy`
- Training labels: `label_train.npy`
- Test data: `DCASE2022_test.npy`
- Test labels: `label_test.npy`

These are intended to be used as input to the `sota_baseline.py` model, based on Singh & Surrey's submission to the 2022 DCASE challenge. Our improvements models include scattering transforms in place of mel spectrograms.

### Scattering Transforms
Download from [Google Drive](https://drive.google.com/drive/folders/1Tc_duS9sVX9e9o62qECTafI2xk5AN-Y7?usp=drive_link)

## Running the Baseline Model

To run the baseline model:
```bash
python sota_baseline.py
```

## Monitoring Training with TensorBoard

The baseline model includes TensorBoard integration for real-time training monitoring. TensorBoard provides interactive visualizations of:
- Training and validation accuracy/loss over time
- Model architecture
- Weight histograms
- Training metrics

To view the TensorBoard dashboard:
```bash
tensorboard --logdir logs/all_models
```

Then open your web browser and navigate to:
```
http://localhost:6006
```

The logs are automatically saved in the `logs/fit` directory with timestamps for each training run.

## Results and Outputs
All training outputs, logs, and results should be saved in the `outputs/` directory. This includes:
- Model checkpoints
- Training logs
- Evaluation metrics
- Visualizations