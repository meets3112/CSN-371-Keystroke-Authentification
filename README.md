# Keystroke Authentication using Machine Learning

This an implemenatation of Continuous authentication by free-text keystroke based on CNN and RNN, 2020. It focuses on behavioral biometric authentication through keystroke dynamics analysis. It processes keystroke data, extracts features, and trains machine learning models to recognize users based on their typing patterns.

## Project Structure

- **key_mapping.py**: Maps all the keys to a unique integer and stores it in a .csv
- **data_conversion.py**: Processes raw keystroke data from the UB Keystroke Dataset and extracts relevant features
- **sliding_window.py**: Prepares data with sliding window of 30 and stores it in .h5 file 
- **model.py**: Contains code to build, train, and evaluate LSTM neural network models
- **test_model.py**: Provides functionality to test trained models and evaluate their performance metrics

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- pandas
- h5py
- matplotlib

### Dataset

University of Buffalo Keystroke dataset was used for training the model. It is not included for privacy concerns.



