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

def process_directory(base_dir, subdirs=['baseline', 'rotation']):
    """Process all files in the given directories."""
    all_data = []
    
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue
            
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(dir_path, file_name)
                try:
                    key_events = read_keystroke_file(file_path)
                    # Add metadata
                    file_data = {
                        'file_name': file_name,
                        'subject_dir': os.path.basename(base_dir),
                        'condition': subdir,
                        'key_events': key_events
                    }
                    all_data.append(file_data)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return all_data

def build_key_mapping(directory_paths):
    """Build a mapping of key names to integer values by scanning all files."""
    key_map = {}
    next_id = 1  
    
    for base_dir in directory_paths:
        for subdir in ['baseline', 'rotation']:
            dir_path = os.path.join(base_dir, subdir)
            if not os.path.exists(dir_path):
                print(f"Directory not found: {dir_path}")
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
                                if key_type not in key_map:
                                    key_map[key_type] = next_id
                                    next_id += 1
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    
    return key_map

def convert_to_dataframe(keystroke_data, key_map):
    """Convert keystroke data to a pandas DataFrame with mapped key values."""
    rows = []
    
    for data in keystroke_data:
        file_name = data['file_name']
        subject = data['subject_dir']
        condition = data['condition']
        
        for key_type, key_action, timestamp in data['key_events']:
            key_code = key_map.get(key_type, -1)  # -1 for unknown keys
            action_code = 1 if key_action == 'KeyDown' else 0  # 1 for down, 0 for up
            
            rows.append({
                'file_name': file_name,
                'subject': subject,
                'condition': condition,
                'key': key_type,
                'key_code': key_code,
                'action': key_action,
                'action_code': action_code,
                'timestamp': timestamp
            })
    
    return pd.DataFrame(rows)

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    input_directories = [
        os.path.join(base_path, "UB_keystroke_dataset/s0"),
        os.path.join(base_path, "UB_keystroke_dataset/s1"),
        os.path.join(base_path, "UB_keystroke_dataset/s2")
    ]
    
    print("Building key mapping...")
    key_map = build_key_mapping(input_directories)
    print(f"Found {len(key_map)} unique keys")
    
    mapping_df = pd.DataFrame([(k, v) for k, v in key_map.items()], columns=['key', 'code'])
    mapping_df.to_csv(os.path.join(base_path, 'key_mapping.csv'), index=False)
    print(f"Key mapping saved to 'key_mapping.csv'")

if __name__ == "__main__":
    main()
