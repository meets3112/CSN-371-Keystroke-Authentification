import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import h5py
import os
import sys
from datetime import datetime
    
from model import load_data

def load_model_and_test(model_path, data_path, window_size=30):
    """
    Load a saved model and test it with validation data
    """
    # Create a text file to save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"test_results_{timestamp}.txt"
    results_file = open(results_filename, "w")
    
    # Function to print to both console and file
    def print_result(message):
        print(message)
        results_file.write(message + "\n")
    
    print_result(f"Test Results for Model: {model_path}")
    print_result(f"Data: {data_path}")
    print_result(f"Window Size: {window_size}")
    print_result(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_result("="*50)
    
    print_result(f"Loading model from: {model_path}")
    try:
        model = keras.models.load_model(model_path)
        print_result("Model loaded successfully")
        
        orig_stdout = sys.stdout
        sys.stdout = results_file
        model.summary()
        sys.stdout = orig_stdout
        print("Model summary saved to results file")
        
    except Exception as e:
        print_result(f"Error loading model: {e}")
        results_file.close()
        return
    
    print_result(f"Loading validation data from: {data_path}")
    try:
        _, _, val0_data, val0_labels, val1_data, val1_labels = load_data(data_path, window_size)
    except Exception as e:
        print_result(f"Error loading data: {e}")
        results_file.close()
        return
    
    print_result("\n\n======= Evaluating model on Class 0 validation data =======")
    if len(val0_data) > 0 and val0_data.size > 0:
        results0 = model.evaluate(val0_data, val0_labels, verbose=1)
        print_result(f"Loss on class 0 data: {results0[0]}, Accuracy: {results0[1] * 100:.2f}%")
        
        predictions0 = model.predict(val0_data)
        
        true_negatives = 0
        false_positives = 0
        
        for i in range(len(predictions0)):
            pred_class = np.argmax(predictions0[i])
            true_class = int(val0_labels[i])
            if pred_class == true_class:
                true_negatives += 1
            else:
                false_positives += 1
        
        print_result(f"True Negatives (correctly predicted class 0): {true_negatives}")
        print_result(f"False Positives (class 0 predicted as class 1): {false_positives}")
        
        print_result("\nSample predictions for class 0:")
        for i in range(min(5, len(predictions0))):
            print_result(f"Sample {i}: Probabilities {predictions0[i]}, True: {val0_labels[i]}, Predicted: {np.argmax(predictions0[i])}")
    else:
        print_result("No Class 0 validation data available")
    
    print_result("\n\n======= Evaluating model on Class 1 validation data =======")
    if len(val1_data) > 0 and val1_data.size > 0:
        results1 = model.evaluate(val1_data, val1_labels, verbose=1)
        print_result(f"Loss on class 1 data: {results1[0]}, Accuracy: {results1[1] * 100:.2f}%")
        
        predictions1 = model.predict(val1_data)
        
        true_positives = 0
        false_negatives = 0
        
        for i in range(len(predictions1)):
            pred_class = np.argmax(predictions1[i])
            true_class = int(val1_labels[i])
            if pred_class == true_class:
                true_positives += 1
            else:
                false_negatives += 1
        
        print_result(f"True Positives (correctly predicted class 1): {true_positives}")
        print_result(f"False Negatives (class 1 predicted as class 0): {false_negatives}")
        
        print_result("\nSample predictions for class 1:")
        for i in range(min(5, len(predictions1))):
            print_result(f"Sample {i}: Probabilities {predictions1[i]}, True: {val1_labels[i]}, Predicted: {np.argmax(predictions1[i])}")
    else:
        print_result("No Class 1 validation data available")
    
    if len(val0_data) > 0 and len(val1_data) > 0:
        print_result("\n\n======= Combined evaluation metrics =======")
        combined_data = np.vstack((val0_data, val1_data))
        combined_labels = np.hstack((val0_labels, val1_labels))
        
        combined_results = model.evaluate(combined_data, combined_labels, verbose=1)
        print_result(f"Overall Loss: {combined_results[0]}, Overall Accuracy: {combined_results[1] * 100:.2f}%")
        
        predictions = model.predict(combined_data)
        pred_classes = np.argmax(predictions, axis=1)
        
        tp = true_positives
        tn = true_negatives
        fp = false_positives
        fn = false_negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Acceptance Rate
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Rejection Rate
        
        # Equal Error Rate approximation (when FAR = FRR)
        eer = (far + frr) / 2
        
        print_result(f"Precision: {precision * 100:.2f}%")
        print_result(f"Recall: {recall * 100:.2f}%")
        print_result(f"F1 Score: {f1_score * 100:.2f}%")
        print_result("\n----- Biometric Security Metrics -----")
        print_result(f"FAR (False Acceptance Rate): {far * 100:.2f}%")
        print_result(f"FRR (False Rejection Rate): {frr * 100:.2f}%") 
        print_result(f"EER (Equal Error Rate approximation): {eer * 100:.2f}%")
        
        plt.figure(figsize=(8, 6))
        cm = np.array([[tn, fp], [fn, tp]])
        
        total_samples = tn + fp + fn + tp
        cm_percent = cm / total_samples * 100
        
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Class 0', 'Class 1'])
        plt.yticks(tick_marks, ['Class 0', 'Class 1'])
        
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f"{cm[i, j]} \n({cm_percent[i, j]:.1f}%)",
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        confusion_matrix_filename = f"confusion_matrix_{timestamp}.png"
        plt.savefig(confusion_matrix_filename)
        print_result(f"Confusion matrix saved to: {confusion_matrix_filename}")
        plt.show()
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot([far], [frr], 'ro', markersize=8, label='Operating Point')
        plt.xlabel('False Acceptance Rate (FAR)')
        plt.ylabel('False Rejection Rate (FRR)')
        plt.title('FAR vs FRR (EER point would be at intersection with diagonal)')
        plt.legend()
        plt.grid(True)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        far_frr_filename = f"far_frr_{timestamp}.png"
        plt.savefig(far_frr_filename)
        print_result(f"FAR vs FRR plot saved to: {far_frr_filename}")
        plt.show()
    
    results_file.close()
    print(f"All results have been saved to: {results_filename}")

if __name__ == "__main__":
    model_path = "1D_1xLSTM32_id1_ws30.h5"
    data_path = "id1_WS30.h5"
    window_size = 30
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        available_files = [f for f in os.listdir('.') if f.endswith('.h5')]
        if available_files:
            print(f"Available .h5 files: {available_files}")
            model_path = input("Enter the name of the model file to use: ")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found.")
        available_files = [f for f in os.listdir('.') if f.endswith('.h5')]
        if available_files:
            print(f"Available .h5 files: {available_files}")
            data_path = input("Enter the name of the data file to use: ")
    
    # Run the testing
    load_model_and_test(model_path, data_path, window_size)
