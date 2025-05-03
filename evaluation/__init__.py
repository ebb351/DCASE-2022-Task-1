"""
DCASE 2022 Task 1 - Evaluation Package

This package contains tools for loading and evaluating models for the DCASE 2022 Task 1
acoustic scene classification challenge.
"""

from .model_loader import (
    load_model_with_custom_layers,
    test_model,
    load_model_file
) 