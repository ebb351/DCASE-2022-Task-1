"""
DCASE2022 Challenge - Acoustic Scene Classification with Scattering Transform Features

This script implements a neural network for acoustic scene classification that uses 
scattering transform features instead of mel-spectrograms for enhanced performance.

Key Features:
- Input: Scattering transform features
- Architecture: Convolutional layers with deeper structure
- Output: 10-class classification (acoustic scenes)
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

def load_data():
    """Load and preprocess the DCASE2022 dataset from scattering transform numpy files."""
    print("🔍 Loading scattering transform data...")
    BASE_PATH = 'data/scattering_transform'

    # Load and concatenate training features and labels
    x_train_1 = np.load(os.path.join(BASE_PATH, 'X_train_part_1.npy'))
    x_train_2 = np.load(os.path.join(BASE_PATH, 'X_train_part_2.npy'))
    y_train_1 = np.load(os.path.join(BASE_PATH, 'y_train_part_1_num.npy'))
    y_train_2 = np.load(os.path.join(BASE_PATH, 'y_train_part_2_num.npy'))

    x_train = np.concatenate([x_train_1, x_train_2], axis=0)
    y_train = np.concatenate([y_train_1, y_train_2], axis=0)

    # Load test features and labels
    x_test = np.load(os.path.join(BASE_PATH, 'X_test.npy'))
    y_test = np.load(os.path.join(BASE_PATH, 'y_test_num.npy'))

    print(f"📊 Train features shape (before split): {x_train.shape}")
    print(f"📊 Train labels shape (before split): {y_train.shape}")
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

def create_datasets(x_train, x_val, x_test, labels_train, labels_val, labels_test, batch_size=64):
    """Create tf.data.Dataset objects for training, validation, and testing."""
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, labels_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, labels_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, labels_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, test_dataset

def create_model(input_shape=(52, 128), num_classes=10):
    """Create the CNN model architecture for scattering transform features."""
    model = models.Sequential()
    model.add(layers.Input(shape=(*input_shape, 1)))

    # First convolution block
    model.add(layers.Conv2D(16, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('tanh'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    # Second convolution block
    model.add(layers.Conv2D(16, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # Third convolution block
    model.add(layers.Conv2D(48, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Fourth convolution block (not sure if we can use this, not 4th block in the original model)
    # model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Activation('relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Dropout(0.35))

    # Global pooling instead of flatten
    model.add(layers.GlobalAveragePooling2D())

    # Dense layer with L2 regularization
    model.add(layers.Dense(100, kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.4))

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
    log_dir = os.path.join("logs", "all_models", "scattering", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Get unique model path
    model_path = get_unique_model_path("scattering")
    
    # Load data
    x_train, x_val, x_test, y_train, y_val, y_test = load_data()
    
    # Create datasets with increased batch size
    batch_size = 64
    train_dataset, val_dataset, test_dataset = create_datasets(
        x_train, x_val, x_test, y_train, y_val, y_test, batch_size=batch_size
    )
    
    # Create and compile model
    print("\n🤖 Creating and compiling model...")
    model = create_model()
    
    # Use a fixed learning rate instead of a scheduler to work with ReduceLROnPlateau
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
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
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train the model with extended epochs
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
    
    # Log test results
    log_test_results(
        model_name="scattering",
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