"""
DCASE2022 Challenge - Acoustic Scene Classification Baseline Model

This script implements a baseline convolutional neural network (CNN) for acoustic scene classification
using mel-spectrogram features. The model architecture is specifically designed for the DCASE2022
challenge dataset.

Key Features:
- Input: Mel-spectrogram features of shape (40, 51)
- Architecture: Three convolutional blocks with increasing complexity
- Output: 10-class classification (acoustic scenes)
- Training: Includes early stopping and model checkpointing
- Evaluation: Provides accuracy metrics and training visualization
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.test_logger import log_test_results, get_unique_model_path

def mobilenet_block(x, filters, kernel_size=(3, 3), strides=(1, 1), alpha=1.0, block_id=0):
    """MobileNet-style depthwise separable conv block."""
    prefix = f'block_{block_id}_'
    
    # Depthwise convolution
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same',
                               use_bias=False, name=prefix + 'dwconv')(x)
    x = layers.BatchNormalization(name=prefix + 'dw_bn')(x)
    x = layers.ReLU(6., name=prefix + 'dw_relu')(x)
    
    # Pointwise convolution
    filters = int(filters * alpha)
    x = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False,
                      name=prefix + 'pwconv')(x)
    x = layers.BatchNormalization(name=prefix + 'pw_bn')(x)
    x = layers.ReLU(6., name=prefix + 'pw_relu')(x)
    
    return x

def load_data():
    """Load and preprocess the DCASE2022 dataset from numpy files."""
    print("🔍 Loading data...")
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    # Load training and testing features and labels
    x_train = np.load(os.path.join(BASE_PATH, 'DCASE2022_train.npy'))
    y_train = np.load(os.path.join(BASE_PATH, 'label_train.npy'))

    x_test = np.load(os.path.join(BASE_PATH, 'DCASE2022_test.npy'))
    y_test = np.load(os.path.join(BASE_PATH, 'label_test.npy'))
    
    print(f"📊 Train features shape: {x_train.shape}")
    print(f"📊 Train labels shape: {y_train.shape}")
    print(f"📊 Test features shape: {x_test.shape}")
    print(f"📊 Test labels shape: {y_test.shape}")
    
    # Split training data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Add channel dimension for Conv2D
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    print(f"📊 Training data shape: {x_train.shape}")
    print(f"📊 Validation data shape: {x_val.shape}")
    print(f"📊 Test data shape: {x_test.shape}")
    print(f"📊 Training labels shape: {y_train.shape}")
    print(f"📊 Validation labels shape: {y_val.shape}")
    print(f"📊 Test labels shape: {y_test.shape}")

    return x_train, x_val, x_test, y_train, y_val, y_test

def create_datasets(x_train, x_val, x_test, labels_train, labels_val, labels_test, batch_size=32):
    """Create tf.data.Dataset objects for training, validation, and testing."""
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, labels_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, labels_val))
    val_dataset = val_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, labels_test))
    test_dataset = test_dataset.batch(batch_size)
    
    return train_dataset, val_dataset, test_dataset

def create_model(input_shape=(40, 51), num_classes=10):
    inputs = tf.keras.Input(shape=(*input_shape, 1))

    # Adjusted filter counts to increase total parameter size to ~16K
    x = mobilenet_block(inputs, filters=32, block_id=1)
    x = layers.Dropout(0.2)(x)
    
    x = mobilenet_block(x, filters=32, block_id=2)
    x = layers.Dropout(0.25)(x)
    
    x = layers.AveragePooling2D(pool_size=(5, 5))(x)
    
    x = mobilenet_block(x, filters=64, block_id=3)
    x = layers.Dropout(0.3)(x)

    x = layers.AveragePooling2D(pool_size=(4, 10))(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(100, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
def plot_training_history(history):
    """Plot and save training history."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create timestamped log directory for TensorBoard
    log_dir = os.path.join("logs", "all_models", "sota_baseline", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Get unique model path
    model_path = get_unique_model_path("sota_baseline")
    
    # Load data
    x_train, x_val, x_test, y_train, y_val, y_test = load_data()
    
    # Create datasets
    batch_size = 32
    train_dataset, val_dataset, test_dataset = create_datasets(
        x_train, x_val, x_test, y_train, y_val, y_test, batch_size=batch_size
    )
    
    # Create and compile model
    print("\n🤖 Creating and compiling model...")
    model = create_model()
    
    # Compile with Adadelta optimizer and categorical crossentropy
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adadelta(),
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    total_params = model.count_params()
    print(f"\n📦 Total parameters: {total_params}")

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    print("\n🚀 Starting training...")
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    print("📊 To view TensorBoard, run: tensorboard --logdir logs/fit")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=callbacks,
        verbose=2
    )
    
    # Evaluate on test set
    print("\n📈 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
    
    # Calculate log loss
    y_pred = model.predict(x_test)
    logloss = tf.keras.losses.categorical_crossentropy(y_test, y_pred)
    logloss = tf.reduce_mean(logloss).numpy()
    
    print(f"\n✅ Test accuracy: {test_acc:.4f}")
    print(f"✅ Test log loss: {logloss:.4f}")
    
    # Log test results to CSV
    log_test_results(
        model_name="sota_baseline",
        test_acc=test_acc,
        test_logloss=logloss,
        model_path=model_path,
        tensorboard_logdir=log_dir
    )
    
    # Plot training history
    print("\n📊 Plotting training history...")
    plot_training_history(history)
    print("\n🎉 Training completed! Results saved in 'training_history.png'")
    print(f"🎉 TensorBoard logs saved to: {log_dir}")
    print(f"🎉 Model saved to: {model_path}")

if __name__ == '__main__':
    main()