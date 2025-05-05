#!/usr/bin/env python3
"""
DCASE 2022 Task 1 - Enhanced Model Evaluation Script

This script loads models using the improved model_loader.py and evaluates them
on the DCASE dataset, providing comprehensive metrics including accuracy, loss, 
precision, recall, and F1 score (overall and per-class).
Results are stored in a structured CSV file for easy analysis.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.model_loader import load_model_with_custom_layers

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Output file for storing evaluation results
RESULTS_CSV = "evaluation_results.csv"

def load_data(data_type='mel_spec'):
    """
    Load the DCASE dataset based on the data type.
    
    Args:
        data_type: 'mel_spec' for mel spectrograms or 'scattering' for scattering transforms
        
    Returns:
        x_test, y_test, y_test_categorical: Test data and labels (both numeric and one-hot encoded)
    """
    print(f"Loading {data_type} test data...")
    
    if data_type == 'mel_spec':
        # Load mel spectrogram test data
        BASE_PATH = 'data/mel_spec'
        x_test = np.load(os.path.join(BASE_PATH, 'data_test.npy'))
        y_test = np.load(os.path.join(BASE_PATH, 'label_test.npy'))
        
        # Add channel dimension
        x_test = np.expand_dims(x_test, axis=-1)
        
    elif data_type == 'scattering':
        # Load scattering transform test data
        BASE_PATH = 'data/scattering_transform'
        x_test = np.load(os.path.join(BASE_PATH, 'X_test.npy'))
        y_test = np.load(os.path.join(BASE_PATH, 'y_test_num.npy'))
        
        # Add channel dimension
        x_test = np.expand_dims(x_test, axis=-1)
        
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    # Convert labels to categorical
    y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=10)
    
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test_categorical.shape}")
    
    return x_test, y_test, y_test_categorical

def evaluate_model(model, x_test, y_test_numeric, y_test_categorical, class_names=None):
    """
    Evaluate a model on test data with comprehensive metrics.
    
    Args:
        model: Keras model to evaluate
        x_test: Test data
        y_test_numeric: Numeric test labels (not one-hot encoded)
        y_test_categorical: One-hot encoded test labels
        class_names: List of class names
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(x_test, verbose=1)
    
    # Calculate loss
    loss = tf.keras.losses.categorical_crossentropy(y_test_categorical, y_pred)
    loss = tf.reduce_mean(loss).numpy()
    
    # Convert to class indices
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate overall metrics
    accuracy_micro = accuracy_score(y_test_numeric, y_pred_classes)
    precision_overall = precision_score(y_test_numeric, y_pred_classes, average='macro')
    recall_overall = recall_score(y_test_numeric, y_pred_classes, average='macro')
    f1_overall = f1_score(y_test_numeric, y_pred_classes, average='macro')

    # Also calculate micro (global) metrics
    precision_micro = precision_score(y_test_numeric, y_pred_classes, average='micro')
    recall_micro = recall_score(y_test_numeric, y_pred_classes, average='micro')
    f1_micro = f1_score(y_test_numeric, y_pred_classes, average='micro')
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_test_numeric, y_pred_classes, average=None)
    recall_per_class = recall_score(y_test_numeric, y_pred_classes, average=None)
    f1_per_class = f1_score(y_test_numeric, y_pred_classes, average=None)
    
    # Calculate per-class accuracy
    accuracy_per_class = []
    for i in range(len(class_names)):
        class_mask = y_test_numeric == i
        if np.sum(class_mask) > 0:
            class_accuracy = accuracy_score(y_test_numeric[class_mask], y_pred_classes[class_mask])
        else:
            class_accuracy = 0.0
        accuracy_per_class.append(class_accuracy)
    
    # Calculate average per-class accuracy
    accuracy_macro = np.mean(accuracy_per_class)
    
    # Print overall results
    print(f"\nOverall Results:")
    print(f"Accuracy (Micro): {accuracy_micro:.4f}")
    print(f"Accuracy (Macro): {accuracy_macro:.4f}")
    print(f"Loss: {loss:.4f}")
    print(f"Precision (macro): {precision_overall:.4f}")
    print(f"Recall (macro): {recall_overall:.4f}")
    print(f"F1 Score (macro): {f1_overall:.4f}")
    print(f"Precision (micro): {precision_micro:.4f}")
    print(f"Recall (micro): {recall_micro:.4f}")
    print(f"F1 Score (micro): {f1_micro:.4f}")
    
    # Print per-class results if class names are provided
    if class_names is not None:
        print("\nPer-Class Results:")
        for i, class_name in enumerate(class_names):
            print(f"{class_name}:")
            print(f"  Accuracy: {accuracy_per_class[i]:.4f}")
            print(f"  Precision: {precision_per_class[i]:.4f}")
            print(f"  Recall: {recall_per_class[i]:.4f}")
            print(f"  F1 Score: {f1_per_class[i]:.4f}")
    
    # Generate confusion matrix if class names are provided
    if class_names is not None:
        cm = confusion_matrix(y_test_numeric, y_pred_classes)
        model_name = model.name if hasattr(model, 'name') and model.name else 'model'
        clean_model_name = model_name.replace(' ', '_').replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = f"confusion_matrices/confusion_matrix_{clean_model_name}_{timestamp}.png"

        fig, ax = plt.subplots(figsize=(14, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(
            cmap='Blues',
            values_format='.0f',
            ax=ax,
            colorbar=True,
            xticks_rotation=45  # 45 is often best for alignment
        )

        # Set title and labels with better spacing
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=20, pad=20)
        ax.set_xlabel('Predicted label', fontsize=16, labelpad=15)
        ax.set_ylabel('True label', fontsize=16, labelpad=15)

        # Make sure tick labels are centered and not cut off
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor", fontsize=13)
        plt.setp(ax.get_yticklabels(), fontsize=13)

        # Adjust value font size for better fit
        for text in ax.texts:
            text.set_fontsize(11)

        fig.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Confusion matrix saved to {output_file}")
    
    # Compile all metrics into a dictionary
    metrics = {
        'accuracy_micro': accuracy_micro,
        'accuracy_macro': accuracy_macro,
        'loss': loss,
        'precision_overall': precision_overall,
        'recall_overall': recall_overall,
        'f1_overall': f1_overall,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
    }
    
    # Add per-class metrics
    if class_names is not None:
        for i, class_name in enumerate(class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
            metrics[f'accuracy_{class_name}'] = accuracy_per_class[i]
    
    return metrics

def save_metrics_to_csv(model_path, data_type, metrics, class_names):
    """
    Save evaluation metrics to a CSV file.
    
    Args:
        model_path: Path to the model file
        data_type: Type of data used for evaluation
        metrics: Dictionary of evaluation metrics
        class_names: List of class names
    """
    # Create a dictionary for the new row
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_filename = os.path.basename(model_path)
    model_name = os.path.splitext(model_filename)[0]
    
    # Initialize row with base metrics
    row = {
        'timestamp': timestamp,
        'model_name': model_name,
        'model_path': model_path,
        'data_type': data_type,
        'accuracy_micro': metrics['accuracy_micro'],
        'accuracy_macro': metrics['accuracy_macro'],
        'loss': metrics['loss'],
        'precision_overall': metrics['precision_overall'],
        'recall_overall': metrics['recall_overall'],
        'f1_overall': metrics['f1_overall'],
        'precision_micro': metrics['precision_micro'],
        'recall_micro': metrics['recall_micro'],
        'f1_micro': metrics['f1_micro'],
    }
    
    # Add per-class metrics in the correct order
    for class_name in class_names:
        # Add precision, recall, f1, and accuracy for each class
        row[f'precision_{class_name}'] = metrics[f'precision_{class_name}']
        row[f'recall_{class_name}'] = metrics[f'recall_{class_name}']
        row[f'f1_{class_name}'] = metrics[f'f1_{class_name}']
        row[f'accuracy_{class_name}'] = metrics[f'accuracy_{class_name}']
    
    # Check if file exists
    file_exists = os.path.isfile(RESULTS_CSV)
    
    # Create DataFrame for the new row
    df = pd.DataFrame([row])
    
    if file_exists:
        try:
            # Read existing CSV file
            existing_df = pd.read_csv(RESULTS_CSV)
            
            # If the existing CSV doesn't have accuracy columns, we need to add them
            # Combine columns from both DataFrames to ensure all are included
            all_columns = list(existing_df.columns) + [col for col in df.columns if col not in existing_df.columns]
            
            # Create a new merged DataFrame with all columns
            merged_df = pd.DataFrame([row], columns=all_columns)
            
            # Append to the existing file while preserving the header
            merged_df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error handling existing CSV: {e}")
            print("Creating a new CSV file instead.")
            df.to_csv(RESULTS_CSV, index=False)
    else:
        # Create a new CSV file
        df.to_csv(RESULTS_CSV, index=False)
    
    print(f"\nResults saved to {RESULTS_CSV}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate DCASE 2022 Task 1 models with comprehensive metrics')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file (.keras)')
    parser.add_argument('--data-type', type=str, default='auto', 
                       choices=['auto', 'mel_spec', 'scattering'],
                       help='Type of data to use for evaluation')
    args = parser.parse_args()
    
    # Class names for DCASE 2022 Task 1
    class_names = [
        'airport',
        'bus',
        'metro',
        'metro_station',
        'park',
        'public_square',
        'shopping_mall',
        'street_pedestrian',
        'street_traffic',
        'tram'
    ]
    
    # Determine model type from filename
    model_file = os.path.basename(args.model)
    model_name = os.path.splitext(model_file)[0]  # Extract model name without extension
    
    if 'all_improvements' in model_file or 'scattering' in model_file:
        model_data_type = 'scattering'
        input_shape = (52, 128)
    else:
        model_data_type = 'mel_spec'
        input_shape = (40, 51)
    
    # Use specified data type or auto-detect
    data_type = args.data_type
    if data_type == 'auto':
        data_type = model_data_type
        print(f"Auto-detected data type: {data_type}")
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = load_model_with_custom_layers(args.model, input_shape=input_shape)
    
    # Set the model name for better identification in plots
    if hasattr(model, 'name'):
        model.name = model_name
    
    # Load test data
    x_test, y_test_numeric, y_test_categorical = load_data(data_type)
    
    # Evaluate the model
    print(f"\nEvaluating model...")
    metrics = evaluate_model(model, x_test, y_test_numeric, y_test_categorical, class_names)
    
    # Save metrics to CSV
    save_metrics_to_csv(args.model, data_type, metrics, class_names)

if __name__ == "__main__":
    main() 