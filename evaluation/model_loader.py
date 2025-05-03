"""
DCASE2022 Task 1: Acoustic Scene Classification Model Loader

This module provides functions for loading TensorFlow/Keras models with custom layers
used in DCASE2022 Task 1, handling .keras format models.

Main Features:
- Custom layer registration for compatibility
- Custom model creation and weight loading
- Support for different architectures used in the challenge
"""

import os
import sys
import importlib.util
import numpy as np
import tensorflow as tf
import tempfile
import shutil
import zipfile
import json
from pathlib import Path

# Add parent directory to path for custom layer imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

# Import h5py for weight loading from h5 files
try:
    import h5py
except ImportError:
    print("Warning: h5py not found. Some weight loading features may not work.")
    h5py = None

# Import custom layers and model creation functions
from models.all_improvements_model import AttentionPooling2D, mobilenet_block
from models.all_improvements_model import create_model as create_improvements_model
from models.mobile_net_model import create_model as create_mobilenet_model

# Import mobile_net_model_2.0.py using importlib since it has a dot in the filename
mobile_net2_path = os.path.join(parent_dir, 'models', 'mobile_net_model_2.0.py')
spec = importlib.util.spec_from_file_location("mobile_net_model_2_0", mobile_net2_path)
mobile_net_model_2_0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mobile_net_model_2_0)
create_mobilenet2_model = mobile_net_model_2_0.create_model

from models.attention_pool_model import create_model as create_attention_pool_model
from models.scattering_model import create_model as create_scattering_model
from models.sota_baseline import create_model as create_sota_baseline_model

# Register custom objects to ensure they can be loaded
CUSTOM_OBJECTS = {
    'AttentionPooling2D': AttentionPooling2D
}

def register_custom_objects():
    """Register custom objects with Keras."""
    for name, obj in CUSTOM_OBJECTS.items():
        tf.keras.utils.get_custom_objects()[name] = obj

# Register custom objects at module import time
register_custom_objects()

def extract_weights_from_keras_file(model, keras_file_path, verbose=True):
    """
    Extract and load weights from a .keras file into a model.
    
    Args:
        model: Keras model to load weights into
        keras_file_path: Path to the .keras file
        verbose: Whether to print details
        
    Returns:
        bool: True if any weights were loaded, False otherwise
    """
    weights_loaded = False
    temp_dir = None
    
    # First, try the direct approach - just load the model and transfer weights
    try:
        if verbose:
            print(f"Attempting direct model loading from {keras_file_path}...")
        
        # Try to load the entire model with custom objects
        source_model = tf.keras.models.load_model(
            keras_file_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )
        
        if verbose:
            print("Successfully loaded full model, transferring weights layer by layer...")
        
        # Transfer weights layer by layer
        target_layers = {layer.name: layer for layer in model.layers}
        loaded_count = 0
        total_target_layers = len([l for l in model.layers if len(l.weights) > 0])
        
        for source_layer in source_model.layers:
            if source_layer.name in target_layers and len(source_layer.weights) > 0:
                target_layer = target_layers[source_layer.name]
                
                # Check if shapes are compatible
                if len(source_layer.weights) == len(target_layer.weights):
                    shape_compatible = True
                    for sw, tw in zip(source_layer.weights, target_layer.weights):
                        if sw.shape != tw.shape:
                            if verbose:
                                print(f"  Shape mismatch for layer {source_layer.name}: {sw.shape} vs {tw.shape}")
                            shape_compatible = False
                            break
                    
                    if shape_compatible:
                        target_layer.set_weights(source_layer.get_weights())
                        loaded_count += 1
                        weights_loaded = True
                        if verbose:
                            print(f"  Loaded weights for layer: {source_layer.name}")
                    else:
                        if verbose:
                            print(f"  Skipping layer {source_layer.name} due to shape mismatch")
                else:
                    if verbose:
                        print(f"  Skipping layer {source_layer.name} due to different number of weights")
        
        if verbose:
            print(f"Loaded weights for {loaded_count}/{total_target_layers} layers with weights")
            
        if loaded_count > 0:
            return True
    except Exception as e:
        if verbose:
            print(f"Direct loading failed: {str(e)}")
    
    # If direct approach failed, try extracting the .keras file
    try:
        # Create a temporary directory to extract the .keras file
        temp_dir = tempfile.mkdtemp()
        
        if verbose:
            print(f"Extracting .keras file to {temp_dir}...")
            
        # Extract the .keras file (ZIP format)
        with zipfile.ZipFile(keras_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # List all files in the extracted directory for debugging
        if verbose:
            print("Extracted file contents:")
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    print(f"  {os.path.join(root, file)}")
        
        # Check for model.weights.h5 (most likely format based on inspection)
        weights_h5_path = os.path.join(temp_dir, 'model.weights.h5')
        if os.path.exists(weights_h5_path):
            if verbose:
                print(f"Found model.weights.h5, trying to load directly...")
            
            try:
                model.load_weights(weights_h5_path)
                if verbose:
                    print("Successfully loaded weights from model.weights.h5")
                return True
            except Exception as e:
                if verbose:
                    print(f"Error loading model.weights.h5: {str(e)}")
        
        # Check for the saved_model.pb file (SavedModel format)
        if os.path.exists(os.path.join(temp_dir, 'saved_model.pb')):
            if verbose:
                print("Found SavedModel format")
                
            # Try to load using SavedModel API
            try:
                source_model = tf.keras.models.load_model(
                    temp_dir,
                    custom_objects=CUSTOM_OBJECTS,
                    compile=False
                )
                
                # Transfer weights layer by layer
                target_layers = {layer.name: layer for layer in model.layers}
                loaded_count = 0
                total_target_layers = len([l for l in model.layers if len(l.weights) > 0])
                
                for source_layer in source_model.layers:
                    if source_layer.name in target_layers:
                        target_layer = target_layers[source_layer.name]
                        
                        # Skip layers with no weights
                        if len(source_layer.weights) == 0:
                            continue
                        
                        # Check if shapes are compatible
                        if len(source_layer.weights) == len(target_layer.weights) and \
                           all(sw.shape == tw.shape for sw, tw in 
                               zip(source_layer.weights, target_layer.weights)):
                            
                            target_layer.set_weights(source_layer.get_weights())
                            loaded_count += 1
                            weights_loaded = True
                            
                            if verbose:
                                print(f"Loaded weights for layer: {source_layer.name}")
                        else:
                            if verbose:
                                print(f"Shape mismatch for layer {source_layer.name}, skipping")
                    
                if verbose:
                    print(f"Loaded weights for {loaded_count}/{total_target_layers} layers with weights")
                
            except Exception as e:
                if verbose:
                    print(f"Error loading from SavedModel: {str(e)}")
        
        # Check for variables directory (checkpoint format)
        variables_dir = os.path.join(temp_dir, 'variables')
        if not weights_loaded and os.path.exists(variables_dir):
            if verbose:
                print("Found variables directory")
            
            # Try to load from checkpoint
            try:
                checkpoint_path = os.path.join(variables_dir, 'variables')
                reader = tf.train.load_checkpoint(checkpoint_path)
                var_to_shape_map = reader.get_variable_to_shape_map()
                
                if verbose:
                    print(f"Checkpoint variables ({len(var_to_shape_map)} total):")
                    for var_name in sorted(list(var_to_shape_map.keys()))[:10]:  # Show first 10
                        print(f"  {var_name}: {var_to_shape_map[var_name]}")
                    if len(var_to_shape_map) > 10:
                        print(f"  ... and {len(var_to_shape_map) - 10} more")
                
                # Map model layers by name
                model_layers = {layer.name: layer for layer in model.layers}
                loaded_count = 0
                
                for layer_name, layer in model_layers.items():
                    if len(layer.weights) == 0:
                        continue
                    
                    layer_weights = []
                    found_all = True
                    
                    # Try to find weights for each parameter in the layer
                    for weight in layer.weights:
                        weight_name = weight.name.split(':')[0]
                        found = False
                        
                        # Try different patterns for checkpoint variable names
                        patterns = [
                            f"{layer_name}/{weight_name.split('/')[-1]}",
                            f"{layer_name}_{weight_name.split('/')[-1]}",
                            weight_name,
                            weight_name.replace('/', '_')
                        ]
                        
                        for pattern in patterns:
                            if pattern in var_to_shape_map:
                                tensor = reader.get_tensor(pattern)
                                if tensor.shape == weight.shape:
                                    layer_weights.append(tensor)
                                    found = True
                                    if verbose:
                                        print(f"  Found weight match: {pattern} -> {weight_name}")
                                    break
                                else:
                                    if verbose:
                                        print(f"  Shape mismatch: {pattern} {tensor.shape} vs {weight.shape}")
                        
                        if not found:
                            if verbose:
                                print(f"  Could not find matching weight for {weight_name}")
                            found_all = False
                            break
                    
                    # Set weights if all were found
                    if found_all and len(layer_weights) == len(layer.weights):
                        layer.set_weights(layer_weights)
                        loaded_count += 1
                        weights_loaded = True
                        
                        if verbose:
                            print(f"Loaded weights for layer: {layer_name}")
                
                if verbose:
                    print(f"Loaded weights for {loaded_count}/{len(model.layers)} layers")
                    
            except Exception as e:
                if verbose:
                    print(f"Error loading from checkpoint: {str(e)}")
    
    except Exception as e:
        if verbose:
            print(f"Error processing .keras file: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return weights_loaded

def identify_model_architecture(model_path):
    """
    Identify the model architecture from a model file path.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        str: Identified model architecture name
    """
    model_file = os.path.basename(model_path)
    
    # Map common model names in filenames to architecture types
    if 'all_improvements' in model_file:
        return 'all_improvements'
    elif 'mobileNet_2.0' in model_file:
        return 'mobileNet_2'
    elif 'mobileNet' in model_file:
        return 'mobileNet'
    elif 'attention_pool' in model_file:
        return 'attention_pool'
    elif 'scattering' in model_file:
        return 'scattering'
    elif 'sota_baseline' in model_file:
        return 'sota_baseline'
    else:
        # Default fallback - extract first part of filename
        return model_file.split('_')[0]

def create_model_architecture(model_name, input_shape=None):
    """
    Create a model architecture based on the model name.
    
    Args:
        model_name: Name of the model architecture
        input_shape: Optional input shape to override defaults
        
    Returns:
        tf.keras.Model: Created model
    """
    # Set appropriate input shape based on model type
    if input_shape is None:
        if model_name in ['scattering', 'all_improvements']:
            input_shape = (52, 128)  # Scattering transform input shape
        else:
            input_shape = (40, 51)   # Mel spectrogram input shape
    
    # Create the appropriate model architecture
    if model_name == 'all_improvements':
        model = create_improvements_model(input_shape=input_shape)
    elif model_name == 'mobileNet':
        # Use the model configuration from mobile_net_model_2.0 for compatibility with saved weights
        # This is because saved mobileNet.keras was likely created with mobile_net_model_2.0.py
        # rather than mobile_net_model.py
        inputs = tf.keras.Input(shape=(*input_shape, 1))
        x = mobilenet_block(inputs, filters=32, block_id=1)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = mobilenet_block(x, filters=64, block_id=2)  # Using 64 filters instead of 32
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(5, 5))(x)
        x = mobilenet_block(x, filters=128, block_id=3)  # Using 128 filters instead of 64
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=(4, 10))(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('tanh')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    elif model_name == 'mobileNet_2':
        model = create_mobilenet2_model(input_shape=input_shape)
    elif model_name == 'attention_pool':
        model = create_attention_pool_model(input_shape=input_shape)
    elif model_name == 'scattering':
        model = create_scattering_model(input_shape=input_shape)
    elif model_name == 'sota_baseline':
        model = create_sota_baseline_model(input_shape=input_shape)
    else:
        print(f"Unknown model type: {model_name}, creating fallback model")
        # Create a simple fallback model
        inputs = tf.keras.Input(shape=(*input_shape, 1))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def load_model_with_custom_layers(model_path, input_shape=None, verbose=True):
    """
    Main function to load a model with custom layers.
    
    This function handles .keras format models and ensures custom layers are properly loaded.
    
    Args:
        model_path: Path to the model file
        input_shape: Optional input shape override
        verbose: Whether to print details
        
    Returns:
        tf.keras.Model: Loaded model with weights
    """
    if verbose:
        print(f"Loading model from {model_path}...")
    
    # Ensure custom objects are registered
    register_custom_objects()
    
    # First try: Direct loading of the model
    try:
        if verbose:
            print("Attempting to load model directly...")
        direct_model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
        if verbose:
            print("Successfully loaded model directly.")
        
        # Configure model for inference
        direct_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return direct_model
        
    except Exception as e:
        if verbose:
            print(f"Direct model loading failed: {str(e)}\nFalling back to reconstruction approach...")
    
    # Identify the model architecture
    model_name = identify_model_architecture(model_path)
    if verbose:
        print(f"Detected model type: {model_name}")
    
    # Normalize input shape
    if input_shape is not None and len(input_shape) == 3:
        input_shape = input_shape[:2]  # Remove channel dimension
    
    # Create the appropriate model architecture
    model = create_model_architecture(model_name, input_shape)
    
    if verbose:
        print(f"Created model architecture for {model_name} with input shape {input_shape}")
        print(f"Model has {len(model.layers)} layers, {model.count_params()} parameters")
    
    # Configure model for inference
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Load weights from .keras file
    weights_loaded = extract_weights_from_keras_file(model, model_path, verbose)
    
    # As a last resort, try standard weights loading
    if not weights_loaded:
        try:
            if verbose:
                print("Attempting standard weights loading...")
            model.load_weights(model_path)
            if verbose:
                print("Standard weights loading succeeded")
            weights_loaded = True
        except Exception as e:
            if verbose:
                print(f"Standard weights loading failed: {str(e)}")
    
    if not weights_loaded and verbose:
        print("WARNING: No weights were loaded, model will use random initialization")
    
    return model

def load_model_file(model_path, custom_objects=None):
    """
    Attempt to load a model file directly using tf.keras.models.load_model.
    
    This is a simpler approach that works when the model file is fully compatible.
    
    Args:
        model_path: Path to the model file
        custom_objects: Dictionary of custom objects
        
    Returns:
        tf.keras.Model or None: Loaded model if successful, None otherwise
    """
    if custom_objects is None:
        custom_objects = CUSTOM_OBJECTS
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Could not load model directly: {str(e)}")
        return None

def test_model(model, x_test, y_test, batch_size=32):
    """
    Test a loaded model on test data.
    
    Args:
        model: Keras model to test
        x_test: Test features
        y_test: Test labels
        batch_size: Batch size for evaluation
        
    Returns:
        tuple: (accuracy, loss)
    """
    # Make sure test data has right dimensions
    if len(x_test.shape) == 3:
        x_test = np.expand_dims(x_test, axis=-1)
    
    # Ensure labels are in categorical format
    if len(y_test.shape) == 1:
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    
    # Compile model if not already compiled
    if not model.compiled_loss:
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")
    
    return accuracy, loss 