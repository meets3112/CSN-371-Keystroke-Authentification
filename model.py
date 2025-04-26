import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import h5py
import math
from keras.callbacks import LearningRateScheduler
import os
import matplotlib.pyplot as plt

def load_data(file_path, window_size=30):
    print("Loading data from:", file_path)
    
    with h5py.File(file_path, 'r') as hdf:
        data = hdf.get('train_data')
        train_data = np.array(data)
        data = hdf.get('train_labels')
        train_labels = np.array(data)
    
    print(f"Loaded data shape: {train_data.shape}, labels shape: {train_labels.shape}")
    
    val0_data = np.array([])
    val1_data = np.array([])
    
    # Extract first 1000 samples for validation
    for i in range(min(1000, len(train_labels))):
        if train_labels[i] == 0:
            val0_data = np.append(val0_data, train_data[i])
        elif train_labels[i] == 1:
            val1_data = np.append(val1_data, train_data[i])
    
    # Create validation labels
    val0_count = int(val0_data.shape[0]/(window_size*6))
    val1_count = int(val1_data.shape[0]/(window_size*6))
    
    val0_labels = np.zeros(val0_count)
    val1_labels = np.ones(val1_count)
    
    if val0_count > 0:
        val0_data = val0_data.reshape(val0_count, window_size, 6)
    if val1_count > 0:
        val1_data = val1_data.reshape(val1_count, window_size, 6)
    
    if len(train_labels) > 1000:
        train_data = train_data[1000:]
        train_labels = train_labels[1000:]
    
    print(f"Training data shape: {train_data.shape}, Training labels shape: {train_labels.shape}")
    print(f"Validation data (class 0) shape: {val0_data.shape}, labels shape: {val0_labels.shape}")
    print(f"Validation data (class 1) shape: {val1_data.shape}, labels shape: {val1_labels.shape}")
    
    return train_data, train_labels, val0_data, val0_labels, val1_data, val1_labels

def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.1
    epochs_drop = 100.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def build_model(input_shape):
    model = keras.models.Sequential()
    model.add(layers.Conv1D(32, 2, activation='relu', input_shape=input_shape))
    model.add(layers.LSTM(32, return_sequences=True))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='softmax'))    
    
    return model

def train_model(model, train_data, train_labels, batch_size=32, epochs=150, validation_split=0.1):

    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optim = keras.optimizers.Adam(learning_rate=0.0)
    metrics = ["accuracy"]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    
    history = model.fit(
        train_data, 
        train_labels,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks_list,
        epochs=epochs,
        shuffle=True
    )
    
    return history

def evaluate_model(model, val_data, val_labels, batch_size=32):
    if len(val_data) == 0 or len(val_labels) == 0:
        print("Validation data or labels are empty. Cannot evaluate.")
        return None
    
    try:
        if len(val_data.shape) != 3:
            print(f"Warning: Validation data has unexpected shape: {val_data.shape}")
            return None
            
        results = model.evaluate(val_data, val_labels, batch_size=batch_size, verbose=1)
        print(f"Loss: {results[0]}, Accuracy: {results[1]}")
        
        predictions = model.predict(val_data)
        print("Sample predictions:")
        for i in range(min(5, len(predictions))):
            print(f"Sample {i}: {predictions[i]}, True Label: {val_labels[i]}, Predicted: {np.argmax(predictions[i])}")
        
        return predictions
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    window_size = 30 
    input_shape = (window_size, 6)
    batch_size = 32
    epochs = 150
    
    # Load data
    data_path = "id1_WS30.h5" 
    
    try:
        train_data, train_labels, val0_data, val0_labels, val1_data, val1_labels = load_data(data_path, window_size)
    except Exception as e:
        print(f"Error loading data: {e}")
        files = os.listdir('.')
        print(f"Files in current directory: {files}")
        print("Please ensure 'id1_WS30.h5' is available")
        return
    
    # Build model
    model = build_model(input_shape)
    model.summary()
    
    # Train model
    history = train_model(model, train_data, train_labels, batch_size, epochs)
    
    # Plot training history
    plot_training_history(history)

    # Save the model
    model_save_path = "1D_1xLSTM32_id1_ws30.h5"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    print("Running on CPU")
    main()
