# DCASE 2022 Task 1 - Evaluation Tools

This directory contains tools for loading and evaluating models for the DCASE 2022 Task 1 acoustic scene classification challenge.

## Contents

- `model_loader.py`: Functions for loading TensorFlow/Keras models with custom layers
- `evaluate_model.py`: Script for evaluating individual models
- `evaluate_all_models.py`: Script for batch evaluation of multiple models

## Model Loader

The model loader provides functions for loading TensorFlow/Keras models with custom layers used in DCASE 2022 Task 1.

### Key Features

- Handles both `.keras` and `.h5` model formats
- Properly loads custom layers like `AttentionPooling2D`
- Supports different model architectures used in the challenge
- Robust weight loading from various serialization formats

### Usage

```python
from evaluation import load_model_with_custom_layers

# Load a model
model = load_model_with_custom_layers("path/to/model.keras")

# Load a model with custom input shape
model = load_model_with_custom_layers("path/to/model.keras", input_shape=(40, 51))
```

## Evaluation Scripts

### Single Model Evaluation

```bash
python evaluation/evaluate_model.py --model-path path/to/model.keras --data-type <mel_spec/or scattering>
```

### Batch Evaluation

```bash
python evaluation/evaluate_all_models.py --models-dir best_models
```

## Available Models and Input Shapes

The loader automatically detects the model type from the filename and uses the appropriate architecture:

| Model Type | Default Input Shape | Description |
|------------|---------------------|-------------|
| `all_improvements` | (52, 128) | Combined improvements model with scattering transforms |
| `attention_pool` | (40, 51) | Model with attention pooling layers |
| `scattering` | (52, 128) | Model using scattering transforms |
| `mobileNet` | (40, 51) | MobileNet-based model |
| `mobileNet_2` | (40, 51) | MobileNet V2-style model |
| `sota_baseline` | (40, 51) | State-of-the-art baseline model |

## Troubleshooting

1. **Shape mismatches**: If you encounter shape mismatches, try providing the correct input shape as an argument to `load_model_with_custom_layers()`.

2. **Random weights**: If the model loaded but predicts with random-looking outputs (e.g., equal probabilities for all classes), the weight loading likely failed. Check if the model file is intact.

3. **Custom layer errors**: Make sure the model architecture matches the weights file. The loader tries to detect the correct architecture based on the filename. 