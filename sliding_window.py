import numpy as np
import os
import pandas as pd
import h5py
import random
from sklearn.utils import shuffle

def read_vectorized_csv(file_path):
    """
    Read the vectorized keystroke data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        feature_data = df[['key1', 'key2', 'dwell1', 'dwell2', 'ud_time', 'dd_time']].values
        return feature_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def process_directory(a_directory, window_size=30):
    """
    Process all CSV files in the given directory and extract features with sliding window.
    Similar to the dataTrans function in the reference file.
    """
    global train_data
    global train_labels
    global sliding_window_data
    
    for i, filename in enumerate(os.listdir(a_directory)):
        if i == 150:  
            break
        
        if filename.endswith('_vectorized.csv'):
            filepath = os.path.join(a_directory, filename)
            print(filename)
            
            label = 1 if filename[:3] == "001" else 0
            
            new_train_data = read_vectorized_csv(filepath)
            
            if new_train_data is None or len(new_train_data) < window_size + 1:
                print(f"Skipping {filename} - not enough data")
                continue
                
            for j in range(0, len(new_train_data)):
                new_train_data[j][0] = new_train_data[j][0] / 104 
                new_train_data[j][1] = new_train_data[j][1] / 104
            
            if label != 1 and new_train_data.shape[0] > 100:
                x = random.randint(0, new_train_data.shape[0] - 101)
                new_train_data = new_train_data[x:x+100]
            
            for k in range(new_train_data.shape[0] - window_size):
                sliding_window_data = np.append(sliding_window_data, new_train_data[k:k+window_size])
            
            temp = np.empty(int(new_train_data.shape[0] - window_size))
            if label == 1:
                print(f"Subject 1 data shape: {str(new_train_data.shape)}")
            temp.fill(label)
            train_labels = np.append(train_labels, temp)
            
    return

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    global train_data, train_labels, sliding_window_data
    train_data = np.array([])
    train_labels = np.array([])
    sliding_window_data = np.array([])
    
    window_size = 30
    
    print(f"Processing data with window size {window_size}...")
    
    # Process directories for each subject
    vectorized_data_paths = [
        os.path.join(base_path, "VectorizedData/s0/baseline"),
        os.path.join(base_path, "VectorizedData/s1/baseline"),
        os.path.join(base_path, "VectorizedData/s2/baseline")
    ]
    
    for data_path in vectorized_data_paths:
        if os.path.exists(data_path):
            print(f"Processing directory: {data_path}")
            process_directory(data_path, window_size)
        else:
            print(f"Directory not found: {data_path}")
    
    # Reshape data for HDF5 format
    train_labels = train_labels.reshape(int(train_labels.shape[0]), 1)
    sliding_window_data = sliding_window_data.reshape(int(train_labels.shape[0]), window_size, 6)
    
    # Shuffle the data
    sliding_window_data, train_labels = shuffle(sliding_window_data, train_labels)
    
    print(f"\nFinal dataset shape: {sliding_window_data.shape}")
    print(f"Labels shape: {train_labels.shape}")
    
    # Save as HDF5 file
    output_file = os.path.join(base_path, f"id1_WS{window_size}.h5")
    with h5py.File(output_file, 'w') as hdf:
        hdf.create_dataset('train_data', data=sliding_window_data)
        hdf.create_dataset('train_labels', data=train_labels)
    
    print(f"\nData saved to {output_file}")

if __name__ == "__main__":
    main()
