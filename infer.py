import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from modelling_seq import AudioSeqModel

def load_model(model_path, hidden_size=128, num_layers=2):
    """Load the trained model from file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioSeqModel(input_size=2, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess_csv(file_path):
    """Preprocess a CSV file for inference."""
    # Read CSV file, skip header rows
    df = pd.read_csv(file_path, skiprows=4)
    
    # Get probe data as features
    probe1 = df.iloc[:, 1].values  # Second column is probe 1
    probe2 = df.iloc[:, 2].values  # Third column is probe 2
    
    # Combine features
    features = np.column_stack((probe1, probe2))
    
    # Convert to tensor (add batch dimension)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    seq_length = torch.tensor([len(features[0])])
    
    return features, seq_length

def predict(model, device, features, seq_length):
    """Make predictions using the model."""
    features = features.to(device)
    
    with torch.no_grad():
        predictions = model(features, seq_length)
    
    return predictions.cpu().numpy()[0]

def visualize_signal_and_prediction(file_path, f_pred, z_pred):
    """Visualize the signal and prediction."""
    # Extract actual values from filename if available
    filename = os.path.basename(file_path)
    import re
    match = re.search(r"output_f=(\d+)_Z=(\d+)_T=(\d+\.\d+)\.csv", filename)
    
    if match:
        f_actual = float(match.group(1))
        z_actual = float(match.group(2))
        title = f"Prediction: f={f_pred:.1f} (Actual: {f_actual}), Z={z_pred:.1f} (Actual: {z_actual})"
    else:
        title = f"Prediction: f={f_pred:.1f}, Z={z_pred:.1f}"
    
    # Read the signal for visualization
    df = pd.read_csv(file_path, skiprows=4)
    time = df.iloc[:, 0].values
    probe1 = df.iloc[:, 1].values
    probe2 = df.iloc[:, 2].values
    
    # Create the plots
    plt.figure(figsize=(12, 8))
    
    # Plot the signals
    plt.subplot(2, 1, 1)
    plt.plot(time, probe1, label='点探针1')
    plt.xlabel('时间 (s)')
    plt.ylabel('总声压 (Pa)')
    plt.title('点探针1 信号')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time, probe2, label='点探针2')
    plt.xlabel('时间 (s)')
    plt.ylabel('总声压 (Pa)')
    plt.title('点探针2 信号')
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the figure
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.splitext(os.path.basename(file_path))[0] + '_prediction.png'
    plt.savefig(os.path.join(output_dir, output_filename))
    
    plt.show()
    
    return match is not None, (f_actual, z_actual) if match else (None, None)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict f and Z from audio signal CSV')
    parser.add_argument('--model', type=str, default='audio_seq_model.pth', help='Path to the model file')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size used in the model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers used in the model')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, device = load_model(args.model, args.hidden_size, args.num_layers)
    
    # Preprocess CSV
    print(f"Processing file: {args.csv}")
    features, seq_length = preprocess_csv(args.csv)
    
    # Make prediction
    print("Making prediction...")
    prediction = predict(model, device, features, seq_length)
    
    # Display results
    f_pred, z_pred = prediction
    print(f"Predicted frequency (f): {f_pred:.2f}")
    print(f"Predicted impedance (Z): {z_pred:.2f}")
    
    # Visualize
    print("Generating visualization...")
    has_actual, (f_actual, z_actual) = visualize_signal_and_prediction(args.csv, f_pred, z_pred)
    
    # Calculate error if actual values are available
    if has_actual:
        f_error = abs(f_pred - f_actual) / f_actual * 100
        z_error = abs(z_pred - z_actual) / z_actual * 100
        print(f"Frequency error: {f_error:.2f}%")
        print(f"Impedance error: {z_error:.2f}%")
    
    print("Done.")

def batch_inference():
    """Run inference on all CSV files in a directory."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch predict f and Z from audio signal CSV files')
    parser.add_argument('--model', type=str, default='audio_seq_model.pth', help='Path to the model file')
    parser.add_argument('--dir', type=str, default='outputs', help='Directory containing CSV files')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size used in the model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers used in the model')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of files to process')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, device = load_model(args.model, args.hidden_size, args.num_layers)
    
    # Get CSV files
    import glob
    csv_files = glob.glob(os.path.join(args.dir, "*.csv"))
    
    if args.limit > 0 and args.limit < len(csv_files):
        import random
        csv_files = random.sample(csv_files, args.limit)
    
    # Results storage
    results = []
    
    # Process each file
    for file_path in csv_files:
        print(f"Processing file: {file_path}")
        features, seq_length = preprocess_csv(file_path)
        prediction = predict(model, device, features, seq_length)
        f_pred, z_pred = prediction
        
        # Extract actual values from filename
        filename = os.path.basename(file_path)
        import re
        match = re.search(r"output_f=(\d+)_Z=(\d+)_T=(\d+\.\d+)\.csv", filename)
        
        if match:
            f_actual = float(match.group(1))
            z_actual = float(match.group(2))
            
            # Calculate error
            f_error = abs(f_pred - f_actual) / f_actual * 100
            z_error = abs(z_pred - z_actual) / z_actual * 100
            
            results.append({
                'file': filename,
                'f_actual': f_actual,
                'f_pred': f_pred,
                'f_error': f_error,
                'z_actual': z_actual,
                'z_pred': z_pred,
                'z_error': z_error
            })
            
            print(f"  Predicted: f={f_pred:.2f}, Z={z_pred:.2f}")
            print(f"  Actual: f={f_actual:.2f}, Z={z_actual:.2f}")
            print(f"  Error: f={f_error:.2f}%, Z={z_error:.2f}%")
        else:
            print(f"  Predicted: f={f_pred:.2f}, Z={z_pred:.2f}")
    
    # Calculate average error
    if results:
        avg_f_error = sum(r['f_error'] for r in results) / len(results)
        avg_z_error = sum(r['z_error'] for r in results) / len(results)
        
        print(f"\nAverage frequency error: {avg_f_error:.2f}%")
        print(f"Average impedance error: {avg_z_error:.2f}%")
        
        # Plot error distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist([r['f_error'] for r in results], bins=10)
        plt.xlabel('Frequency Error (%)')
        plt.ylabel('Count')
        plt.title(f'Frequency Error: Avg={avg_f_error:.2f}%')
        
        plt.subplot(1, 2, 2)
        plt.hist([r['z_error'] for r in results], bins=10)
        plt.xlabel('Impedance Error (%)')
        plt.ylabel('Count')
        plt.title(f'Impedance Error: Avg={avg_z_error:.2f}%')
        
        plt.tight_layout()
        plt.savefig('batch_inference_results.png')
        plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        sys.argv.pop(1)  # Remove the --batch argument
        batch_inference()
    else:
        main()
