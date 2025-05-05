# Machine Listening Final Project: DCASE 2022 Task 1

## Overview
This project implements a baseline model for DCASE 2022 Task 1: Acoustic Scene Classification with Multiple Devices. The goal is to classify acoustic scenes using audio recordings from multiple devices, addressing the challenge of device mismatch in acoustic scene classification.

## Project Structure
```
.
├── data/                    # Dataset storage (git-ignored)
├── best_models/             # Saved best model weights (.keras files)
├── confusion_matrices/      # Confusion matrix visualizations for each model
├── evaluation/              # Evaluation scripts and related files
├── models/                  # Model implementations and saved models
├── logs/                    # Tensorboard logs (git-ignored)
├── preprocessing/           # Scripts for scattering transform creation
├── utils/                   # Utility scripts
└── evaluation_results.csv   # Comprehensive evaluation metrics for all models
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation
1. Clone this repository:
```bash
git clone https://github.com/ebb351/DCASE-2022-Task-1.git
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
Download from [Zenoob](https://zenodo.org/records/6337421)

### Mel Spectrograms
Download from [Google Drive](https://drive.google.com/drive/folders/1m4in9I8e7DtPnJBLo7CYKqONKVYUwmr6)

Relevant Files:
- Training data: `DCASE2022_train.npy`
- Training labels: `label_train.npy`
- Test data: `DCASE2022_test.npy`
- Test labels: `label_test.npy`


### Scattering Transforms
Download from [Google Drive](https://drive.google.com/drive/folders/1Tc_duS9sVX9e9o62qECTafI2xk5AN-Y7?usp=drive_link)

Relevant Files:
- `X_train_part1.npy`, `X_train_part2.npy`: Training data, split into two parts (concatenate for full training set)
- `y_train_part1_str.npy`, `y_train_part2_str.npy`: Training labels (string class names), split into two parts
- `y_train_part1_num.npy`, `y_train_part2_num.npy`: Training labels (numeric class indices), split into two parts
- `X_test.npy`: Test data
- `y_test_str.npy`: Test labels (string class names)
- `y_test_num.npy`: Test labels (numeric class indices)
- `X_eval.npy`: Evaluation data (for validation or leaderboard)
- `X_eval_filenames.npy`: Filenames corresponding to evaluation data

Notes:
- Training data and labels are split into two parts; the training scripts automatically concatenate them.
- Files ending with `_str` contain string class names (e.g., 'airport', 'park').
- Files ending with `_num` contain numeric class indices (e.g., 0, 1, 2, ...).

## Data Storage Instructions

After downloading the datasets, store them in the following directories:
- Mel spectrogram files: `data/mel_spec/`
- Scattering transform files: `data/scattering_transform/`

Ensure that the file names and directory paths match those expected by the training scripts for correct loading and processing.

## Running the Models

The project includes several model implementations in the `models/` directory:
- `sota_baseline.py`: State-of-the-art baseline model
- `attention_pool_model.py`: Model with attention pooling
- `scattering_model.py`: Model using scattering transforms
- `mobile_net_model.py`: MobileNet-based architecture
- `all_improvements_model.py`: Model incorporating all improvements

To run any of the training scripts:
```bash
python models/<model_name>.py
```

Each training run will:
- Save the best model weights (by highest validation accuracy) to a `.keras` file
- Log training metrics to TensorBoard
- Record test results in `test_results_log.csv` (git-ignored)

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

## Model Evaluation

This project includes comprehensive model evaluation capabilities with detailed metrics:

### Key Metrics
- Categorical cross-entropy loss
- Accuracy (per-class and overall)
- Precision (per-class and overall)
- Recall (per-class and overall)
- F1 Score (per-class and overall)

### Running Evaluation

To evaluate a single model:
```bash
python evaluate_model.py --model path/to/model.keras --data-type <scattering/mel_spec>
```

To evaluate all models in the best_models directory:
```bash
python evaluate_all_models.py
```

### Evaluation Results
All evaluation results are stored in:
- `evaluation_results.csv`: A comprehensive CSV file with all metrics for all evaluated models
- `confusion_matrices/`: Directory containing confusion matrix visualizations for each model
- Individual text files with basic metrics for each model (for backward compatibility)

The CSV file provides an easy way to compare model performance across different architectures and track improvements over time.