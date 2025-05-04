"""
DCASE2022 Challenge - Acoustic Scene Classification Model with Combined Improvements

This script implements a neural network for acoustic scene classification that combines
three key improvements:
1. Scattering transforms as input features (instead of log-mel spectrograms)
2. Attention pooling (instead of average pooling)
3. MobileNet blocks (depthwise separable convolutions instead of standard Conv2D)

The model architecture is based on the SOTA baseline but incorporates these improvements
for enhanced performance on the DCASE2022 challenge dataset.

Key Features:
- Input: Scattering transform features
- Architecture: MobileNet blocks with attention pooling
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

# Import utility functions
from utils.model_utils import get_unique_model_path, log_test_results

class AttentionPooling2D(layers.Layer):
    """
    Custom attention pooling layer for acoustic scene classification.
    
    This layer implements a multi-path attention mechanism that combines:
    1. Channel attention: Learns important frequency bands
    2. Frequency attention: Captures vertical patterns in spectrograms
    3. Time attention: Captures horizontal patterns in spectrograms
    
    The layer first applies average pooling, then uses learned attention weights
    to focus on important regions. A learnable mixing factor (alpha) controls
    the balance between attention-weighted and original features.
    
    Attributes:
        pool_size (tuple): Pooling window size (height, width)
        channel_attn: Channel attention convolution
        spatial_attn: Spatial attention convolution
        freq_attn: Frequency-specific attention
        time_attn: Time-specific attention
        alpha: Learnable mixing factor
    """
    
    def __init__(self, pool_size=(2, 2), **kwargs):
        """
        Initialize attention pooling layer.
        
        Args:
            pool_size (tuple): Pooling window size (height, width)
            **kwargs: Additional layer arguments
        """
        super().__init__(**kwargs)
        self.pool_size = pool_size
        
        # Learnable mixing factor (initialized to 0.7)
        self.alpha = None
        self.channel_attn = None
        self.spatial_attn = None
        self.freq_attn = None
        self.time_attn = None
        self.bn = None
        self.bn_freq = None
        self.bn_time = None
    
    def build(self, input_shape):
        """Build the layer and initialize weights."""
        # Channel attention: Learns important frequency bands
        self.channel_attn = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=1, 
            activation='relu', 
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )
        
        # Spatial attention: Learns important regions
        self.spatial_attn = tf.keras.layers.Conv2D(
            filters=1, 
            kernel_size=3, 
            padding='same', 
            activation=None, 
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )
        
        # Frequency attention: Vertical patterns in spectrograms
        self.freq_attn = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 1),  # Vertical kernel
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )
        
        # Time attention: Horizontal patterns in spectrograms
        self.time_attn = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(1, 3),  # Horizontal kernel
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )
        
        # Batch normalization for each attention pathway
        self.bn = tf.keras.layers.BatchNormalization()
        self.bn_freq = tf.keras.layers.BatchNormalization()
        self.bn_time = tf.keras.layers.BatchNormalization()
        
        # Learnable mixing factor (initialized to 0.7)
        self.alpha = tf.Variable(0.7, trainable=True, dtype=tf.float32)
        
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        """
        Forward pass of the attention pooling layer.
        
        Processing steps:
        1. Average pooling to reduce spatial dimensions
        2. Apply channel, frequency, and time attention
        3. Combine attention pathways
        4. Generate spatial attention weights
        5. Apply attention and mix with original features
        
        Args:
            inputs (tf.Tensor): Input tensor (batch, height, width, channels)
            training (bool): Training mode flag
            
        Returns:
            tf.Tensor: Attention-weighted pooled features
        """
        # Initial average pooling
        pooled = tf.nn.avg_pool2d(inputs, ksize=self.pool_size, strides=self.pool_size, padding='VALID')
        
        # Channel attention pathway
        x = self.channel_attn(pooled)
        x = self.bn(x, training=training)
        
        # Frequency attention pathway
        x_freq = self.freq_attn(pooled)
        x_freq = self.bn_freq(x_freq, training=training)
        
        # Time attention pathway
        x_time = self.time_attn(pooled)
        x_time = self.bn_time(x_time, training=training)
        
        # Combine attention pathways
        x = x + x_freq + x_time
        
        # Generate spatial attention weights
        attn_logits = self.spatial_attn(x)
        
        # Normalize attention weights with softmax
        batch_size = tf.shape(attn_logits)[0]
        height = tf.shape(attn_logits)[1]
        width = tf.shape(attn_logits)[2]
        
        attn_flat = tf.reshape(attn_logits, [batch_size, height * width, 1])
        attn_weights = tf.nn.softmax(attn_flat, axis=1)
        attn_map = tf.reshape(attn_weights, [batch_size, height, width, 1])
        
        # Apply attention and mix with original features
        weighted_features = pooled * attn_map
        alpha_sigmoid = tf.sigmoid(self.alpha)  # Constrain between 0 and 1
        mixed_features = alpha_sigmoid * weighted_features + (1-alpha_sigmoid) * pooled
        
        return mixed_features
    
    def compute_output_shape(self, input_shape):
        """Compute output shape after pooling."""
        return (
            input_shape[0],
            input_shape[1] // self.pool_size[0],
            input_shape[2] // self.pool_size[1],
            input_shape[3]
        )
        
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({"pool_size": self.pool_size})
        return config

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
    """Create model with MobileNet blocks and attention pooling for scattering transform features."""
    inputs = tf.keras.Input(shape=(*input_shape, 1))
    
    # First MobileNet block (replacement for C1)
    x = mobilenet_block(inputs, filters=32, block_id=1)
    x = layers.Dropout(0.3)(x)
    
    # Second MobileNet block (replacement for C2)
    x = mobilenet_block(x, filters=64, block_id=2)
    x = layers.Dropout(0.3)(x)

    # P1: Attention Pooling instead of AveragePooling2D
    x = AttentionPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Third MobileNet block (replacement for C3)
    x = mobilenet_block(x, filters=128, block_id=3)
    x = layers.Dropout(0.3)(x)

    # P2: Attention Pooling instead of AveragePooling2D
    x = AttentionPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.3)(x)
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layer with L2 regularization
    x = layers.Dense(100, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('tanh')(x)
    x = layers.Dropout(0.3)(x)

    # Classification + softmax (10 units)
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
    log_dir = os.path.join("logs", "all_models", "all_improvements", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Get unique model path
    model_path = get_unique_model_path("all_improvements")
    
    # Load data
    x_train, x_val, x_test, y_train, y_val, y_test = load_data()
    
    # Create datasets
    batch_size = 64
    train_dataset, val_dataset, test_dataset = create_datasets(
        x_train, x_val, x_test, y_train, y_val, y_test, batch_size=batch_size
    )
    
    # Create and compile model
    print("\n🤖 Creating and compiling model...")
    model = create_model()
    
    # Compile with Adam optimizer and categorical crossentropy
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
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
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
        model_name="all_improvements",
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