import os
import csv
from datetime import datetime

def log_test_results(model_name, test_acc, test_logloss, model_path, tensorboard_logdir):
    """Log test results to a CSV file with model name and timestamp."""
    logfile = 'test_results_log.csv'
    
    # Prepare log data
    log_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'test_accuracy': test_acc,
        'test_logloss': test_logloss,
        'model_path': model_path,
        'tensorboard_logdir': tensorboard_logdir
    }
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.isfile(logfile)
    
    # Write to CSV
    with open(logfile, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    
    print(f"📝 Test results logged to {logfile}")

def get_unique_model_path(model_name):
    """Generate a unique model path based on model name and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"{model_name}_{timestamp}.keras") 