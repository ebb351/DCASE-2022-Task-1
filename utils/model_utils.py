"""
Utility functions for model training and evaluation in the DCASE2022 challenge.

This module contains helper functions used across different model implementations.
"""

import os
from datetime import datetime

print("✅ Successfully imported model_utils module")

def get_unique_model_path(model_name):
    """
    Generate a unique path for saving a model using the model name and current timestamp.
    
    Args:
        model_name (str): The name of the model
        
    Returns:
        str: A unique path for saving the model
    """
    # Create models directory if it doesn't exist
    os.makedirs("models/saved", exist_ok=True)
    
    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create a unique path
    model_path = os.path.join("models/saved", f"{model_name}_{timestamp}.keras")
    
    return model_path

def log_test_results(model_name, test_acc, test_logloss, model_path, tensorboard_logdir):
    """
    Log test results to a CSV file.
    
    Args:
        model_name (str): The name of the model
        test_acc (float): Test accuracy
        test_logloss (float): Test log loss
        model_path (str): Path where the model is saved
        tensorboard_logdir (str): Path to TensorBoard logs
    """
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Path to results file
    results_file = os.path.join("results", "model_results.csv")
    
    # Check if the file exists
    file_exists = os.path.isfile(results_file)
    
    # Write header if file doesn't exist
    with open(results_file, 'a') as f:
        if not file_exists:
            f.write("model_name,test_accuracy,test_logloss,model_path,tensorboard_logdir,timestamp\n")
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write results
        f.write(f"{model_name},{test_acc:.6f},{test_logloss:.6f},{model_path},{tensorboard_logdir},{timestamp}\n")
    
    print(f"✅ Results logged to {results_file}") 