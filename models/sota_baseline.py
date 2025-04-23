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
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

def load_data():
    """Load and preprocess the DCASE2022 dataset from numpy mel files."""
    print("🔍 Loading data...")
    BASE_PATH = 'data'
    
    # Load training features and labels
    x_train = np.load(os.path.join(BASE_PATH, 'data_train.npy'))
    y_train = np.load(os.path.join(BASE_PATH, 'label_train.npy'))
    
    # Load test features and labels
    x_test = np.load(os.path.join(BASE_PATH, 'data_test.npy'))
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
    """Create the CNN model architecture matching Singh Surrey's DCASE2022 individual model."""
    model = models.Sequential()
    
    # C1: Convolution + BN + tanh
    model.add(layers.Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=(*input_shape, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('tanh'))
    
    # C2: Convolution + BN + ReLU
    model.add(layers.Conv2D(16, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # P1: Average Pooling (5x5)
    model.add(layers.AveragePooling2D(pool_size=(5, 5)))
    
    # C3: Convolution + BN + tanh
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('tanh'))

    # P2: Average Pooling (4x10)
    model.add(layers.AveragePooling2D(pool_size=(4, 10)))
    
    # Flatten before dense layers
    model.add(layers.Flatten())
    
    # Dense + tanh (100 units)
    model.add(layers.Dense(100, activation='tanh'))

    # Classification + softmax (10 units)
    model.add(layers.Dense(num_classes, activation='softmax'))
    
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
    log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Load data
    x_train, x_val, x_test, y_train, y_val, y_test = load_data()
    
    # Create datasets
    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    
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
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,  # Log histograms every epoch
            write_graph=True,  # Log the model graph
            write_images=True,  # Log model weights as images
            update_freq='epoch'  # Log metrics at the end of each epoch
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # Stop if validation loss doesn't improve for 10 epochs
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,  # Stop if validation accuracy doesn't improve for 5 epochs
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
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
    
    # Plot training history
    print("\n📊 Plotting training history...")
    plot_training_history(history)
    print("\n🎉 Training completed! Results saved in 'training_history.png'")
    print(f"🎉 TensorBoard logs saved to: {log_dir}")

if __name__ == '__main__':
    main()