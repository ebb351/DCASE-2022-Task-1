"""
DCASE2022 Challenge - Acoustic Scene Classification Baseline Model with Attention Pooling

This script implements a convolutional neural network (CNN) for acoustic scene classification
using mel-spectrogram features with attention pooling. The model architecture is specifically 
designed for the DCASE2022 challenge dataset.

Key Features:
- Input: Mel-spectrogram features of shape (40, 51)
- Architecture: Three convolutional blocks with attention pooling
- Output: 10-class classification (acoustic scenes)
- Training: Includes early stopping and model checkpointing
- Evaluation: Provides accuracy metrics and training visualization

Author: [Your Name]
Date: [Current Date]
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

class AttentionPooling2D(layers.Layer):
    """
    Custom attention pooling layer that learns to focus on important regions of the input.
    
    This layer combines average pooling with learned attention weights to dynamically
    focus on the most relevant parts of the input feature maps. The attention weights
    are learned during training, allowing the model to adaptively pool features based
    on their importance.
    
    Attributes:
        pool_size (tuple): Size of the pooling window (height, width)
        attention_weights (tf.Variable): Learned weights for attention mechanism
    """
    
    def __init__(self, pool_size=(2, 2), **kwargs):
        """
        Initialize the AttentionPooling2D layer.
        
        Args:
            pool_size (tuple): Size of the pooling window (height, width)
            **kwargs: Additional keyword arguments passed to the parent class
        """
        super().__init__(**kwargs)
        self.pool_size = pool_size
        
        # Simplified attention mechanism with L2 regularization to prevent overfitting
        self.channel_attn = tf.keras.layers.Conv2D(
            filters=32,  # Reduced from 64
            kernel_size=1, 
            activation='relu', 
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)  # Add L2 regularization
        )
        
        # Spatial attention with regularization
        self.spatial_attn = tf.keras.layers.Conv2D(
            filters=1, 
            kernel_size=3, 
            padding='same', 
            activation=None, 
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)  # Add L2 regularization
        )
        
        # Add batch normalization for better training stability
        self.bn = tf.keras.layers.BatchNormalization()
        
    def call(self, inputs, training=None):
        """
        Forward pass of the layer.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, height, width, channels)
            training (bool): Whether the layer is in training mode
            
        Returns:
            tf.Tensor: Pooled and attention-weighted output tensor
        """
        # First do the basic average pooling
        pooled = tf.nn.avg_pool2d(inputs, ksize=self.pool_size, strides=self.pool_size, padding='VALID')
        
        # Enhanced attention mechanism
        # 1. Project to higher dimensional space for better representation
        x = self.channel_attn(pooled)
        x = self.bn(x, training=training)
        
        # 2. Generate spatial attention weights
        attn_logits = self.spatial_attn(x)
        
        # 3. Apply softmax over spatial dimensions (H,W) to normalize attention
        # Reshape to (batch, height*width, 1)
        batch_size = tf.shape(attn_logits)[0]
        height = tf.shape(attn_logits)[1]
        width = tf.shape(attn_logits)[2]
        
        attn_flat = tf.reshape(attn_logits, [batch_size, height * width, 1])
        attn_weights = tf.nn.softmax(attn_flat, axis=1)  # Softmax over spatial dimensions
        
        # Reshape back to (batch, height, width, 1)
        attn_map = tf.reshape(attn_weights, [batch_size, height, width, 1])
        
        # Apply attention weights to pooled features
        weighted_features = pooled * attn_map
        
        # Mix with original pooled features to avoid over-specialization
        alpha = 0.7  # Mixing factor
        mixed_features = alpha * weighted_features + (1-alpha) * pooled
        
        return mixed_features
    
    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1] // self.pool_size[0],
            input_shape[2] // self.pool_size[1],
            input_shape[3]
        )
        
    def get_config(self):
        config = super().get_config()
        config.update({"pool_size": self.pool_size})
        return config

def load_data():
    """Load and preprocess the DCASE2022 dataset from numpy mel files."""
    print("🔍 Loading data...")
    BASE_PATH = 'data/mel_spec'
    
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
    """Create the CNN model architecture with attention pooling."""
    model = models.Sequential()
    
    # C1: Convolution + BN + tanh
    model.add(layers.Conv2D(16, kernel_size=(3, 3), padding='same', input_shape=(*input_shape, 1),
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)))  # Add L2 regularization
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('tanh'))
    model.add(layers.Dropout(0.4))  # Increased dropout from 0.3 to 0.4
    
    # C2: Convolution + BN + ReLU
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)))  # Add L2 regularization
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.4))  # Increased dropout from 0.3 to 0.4
    
    # P1: Attention Pooling
    model.add(AttentionPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))  # Increased dropout significantly after attention pooling
    
    # C3: Convolution + BN + tanh
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)))  # Add L2 regularization
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('tanh'))
    model.add(layers.Dropout(0.4))  # Increased dropout from 0.3 to 0.4
    
    # P2: Attention Pooling
    model.add(AttentionPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.5))  # Increased dropout significantly after attention pooling
    
    # Flatten before dense layers
    model.add(layers.Flatten())
    
    # Dense + tanh (32 units)
    model.add(layers.Dense(32, activation='tanh',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)))  # Add L2 regularization
    model.add(layers.Dropout(0.5))  # Increased dropout from 0.3 to 0.5
    
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
    log_dir = os.path.join("logs", "all_models", "attention_pool", datetime.now().strftime("%Y%m%d-%H%M%S"))
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
    
    # Compile with Adam optimizer and categorical crossentropy
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduced learning rate from 0.001 to 0.0005
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
            patience=15,  # Increased patience from 10 to 15
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,  # Increased patience from 5 to 10
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Add learning rate reduction on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=0.00001
        )
    ]
    
    # Train the model
    print("\n🚀 Starting training...")
    print(f"📊 TensorBoard logs will be saved to: {log_dir}")
    print("📊 To view TensorBoard, run: tensorboard --logdir logs/fit")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
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