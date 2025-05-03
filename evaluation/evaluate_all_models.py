#!/usr/bin/env python3
"""
DCASE 2022 Task 1 - Batch Model Evaluation Script

This script evaluates all models in the best_models directory, measuring
comprehensive metrics and storing results in a structured CSV file.
"""

import os
import subprocess
import argparse
import glob
from tqdm import tqdm
import sys

# Add parent directory to path to import evaluate_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.evaluate_model import main as evaluate_model_main

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate all models in best_models directory')
    parser.add_argument('--models-dir', type=str, default='best_models',
                       help='Directory containing models to evaluate')
    args = parser.parse_args()
    
    # Get all model files in directory
    model_files = glob.glob(os.path.join(args.models_dir, '*.keras'))
    model_files.extend(glob.glob(os.path.join(args.models_dir, '*.h5')))
    
    if not model_files:
        print(f"No model files found in {args.models_dir}")
        return
    
    print(f"Found {len(model_files)} model files to evaluate")
    
    # Evaluate each model
    for model_file in tqdm(model_files, desc="Evaluating models"):
        print(f"\n{'='*50}")
        print(f"Evaluating {os.path.basename(model_file)}")
        print(f"{'='*50}")
        
        # Run evaluation script for this model
        try:
            # Use the correct path to evaluate_model.py
            cmd = ['python3', os.path.join('evaluation', 'evaluate_model.py'), '--model', model_file]
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {model_file}: {str(e)}")
    
    print("\nAll models evaluated. Results stored in evaluation_results.csv")

if __name__ == "__main__":
    main() 