import os
import pandas as pd
import numpy as np
from collections import defaultdict

def read_keystroke_file(file_path):
    """Read a keystroke file and extract key events."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    key_events = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            key_type = parts[0]
            key_action = parts[1]
            timestamp = int(parts[2])
            key_events.append((key_type, key_action, timestamp))
    
    return key_events

def extract_features(key_events):
    """
    Extract features from keystroke data using the logic from convert.py:
    - Dwell time (KeyUp - KeyDown) for each key
    - DownDown time (KeyDown_i - KeyDown_j) for adjacent keys
    - UpDown time (KeyDown_j - KeyUp_i) for adjacent keys
    """
    key_events.sort(key=lambda x: x[2])
    
    down_events = []
    up_events = []
    
    for key_type, action, timestamp in key_events:
        if action == 'KeyDown':
            down_events.append((key_type, action, timestamp))
        elif action == 'KeyUp':
            up_events.append((key_type, action, timestamp))
    
    matched_keys = []
    
    for down_event in down_events:
        key_type, _, down_timestamp = down_event
        
        for up_event in up_events:
            up_key_type, _, up_timestamp = up_event
            
            if key_type == up_key_type and up_timestamp > down_timestamp:
                matched_keys.append((key_type, down_timestamp, up_timestamp))
                up_events.remove(up_event)  
                break
    
    vectors = []
    for i in range(len(matched_keys) - 1):
        key1, key1_down, key1_up = matched_keys[i]
        key2, key2_down, key2_up = matched_keys[i + 1]
        
        dwell1 = (key1_up - key1_down) / 1000
        dwell2 = (key2_up - key2_down) / 1000
        
        dd_time = (key2_down - key1_down) / 1000
        
        ud_time = (key2_down - key1_up) / 1000
        
        vector = {
            'key1': key1,
            'key2': key2,
            'dwell1': dwell1,
            'dwell2': dwell2,
            'dd_time': dd_time,
            'ud_time': ud_time
        }
        vectors.append(vector)
    
    return vectors

def vectorize_data(features, adjacency_features):
    """
    Convert extracted features to a vectorized format.
    This function is no longer needed as extract_features now directly returns vectors.
    """
    return features  

def process_directory(base_dir, output_dir, key_map, subdirs=['baseline', 'rotation']):
    """Process all files in the given directories and extract features."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
        
        subdir_output = os.path.join(output_dir, subdir)
        os.makedirs(subdir_output, exist_ok=True)
        
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(dir_path, file_name)
                try:
                    key_events = read_keystroke_file(file_path)
                    
                    vectors = extract_features(key_events)
                    
                    for vector in vectors:
                        vector['key1_id'] = key_map.get(vector['key1'], -1)
                        vector['key2_id'] = key_map.get(vector['key2'], -1)
                    
                    for vector in vectors:
                        vector['key1'], vector['key1_id'], vector['key2'], vector['key2_id'] = \
                            vector['key1_id'], vector['key1'], vector['key2_id'], vector['key2']
                    
                    for vector in vectors:
                        vector['key1_name'] = vector.pop('key1_id')
                        vector['key2_name'] = vector.pop('key2_id')
                    
                    output_file = os.path.join(subdir_output, f"{os.path.splitext(file_name)[0]}_vectorized.csv")
                    
                    df = pd.DataFrame(vectors)
                    df.to_csv(output_file, index=False)
                    
                    print(f"Processed {file_path} -> {output_file}")
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return

def load_key_mapping(file_path):
    """Load key mapping from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return dict(zip(df['key'], df['code']))
    except FileNotFoundError:
        print(f"Key mapping file not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading key mapping: {e}")
        return None

def extract_key_mapping(directory_paths, subdirs=['baseline', 'rotation']):
    """
    Build a mapping of key names to integer values by scanning all files.
    Returns a dictionary with key types as keys and integer IDs as values.
    """
    unique_keys = set()
    
    for base_dir in directory_paths:
        for subdir in subdirs:
            dir_path = os.path.join(base_dir, subdir)
            if not os.path.exists(dir_path):
                continue
                
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(dir_path, file_name)
                    try:
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                        
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                key_type = parts[0]
                                unique_keys.add(key_type)
                    except Exception as e:
                        print(f"Error scanning {file_path}: {e}")
    
    key_map = {key: i+1 for i, key in enumerate(sorted(unique_keys))}
    return key_map

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    input_directories = [
        os.path.join(base_path, "UB_keystroke_dataset/s0"),
        os.path.join(base_path, "UB_keystroke_dataset/s1"),
        os.path.join(base_path, "UB_keystroke_dataset/s2")
    ]
    
    output_directory = os.path.join(base_path, "VectorizedData")
    os.makedirs(output_directory, exist_ok=True)
    
    key_mapping_file = os.path.join(base_path, "key_mapping.csv")
    
    key_map = load_key_mapping(key_mapping_file)
    
    if key_map is None:
        print("Key mapping file not found or invalid. Creating new mapping...")
        key_map = extract_key_mapping(input_directories)
        mapping_df = pd.DataFrame([(k, v) for k, v in key_map.items()], columns=['key', 'code'])
        mapping_df.to_csv(key_mapping_file, index=False)
        print(f"Created new key mapping with {len(key_map)} unique keys")
    else:
        print(f"Loaded key mapping with {len(key_map)} keys from {key_mapping_file}")
    
    for input_dir in input_directories:
        print(f"Processing directory: {input_dir}")
        subject_name = os.path.basename(input_dir)
        subject_output_dir = os.path.join(output_directory, subject_name)
        process_directory(input_dir, subject_output_dir, key_map)
    
    print("Feature extraction complete. Vectorized data saved to:", output_directory)

if __name__ == "__main__":
    main()
