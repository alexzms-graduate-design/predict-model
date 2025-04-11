import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import glob
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
import argparse

class AudioSequenceDataset(Dataset):
    def __init__(self, data_dir, max_seq_length=6000, noise_lambda=0.0):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.noise_lambda = noise_lambda # Store noise lambda
        self.file_paths = []
        self.targets = []
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        for file_path in csv_files:
            # Extract f and Z from filename
            filename = os.path.basename(file_path)
            match = re.search(r"output_f=(\d+)_Z=(\d+)_T=(\d+\.\d+)\.csv", filename)
            if match:
                f = float(match.group(1))
                Z = float(match.group(2))
                
                self.file_paths.append(file_path)
                self.targets.append((f, Z))
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        target = self.targets[idx]
        
        # Read CSV file, skip header rows
        df = pd.read_csv(file_path, skiprows=4)
        
        # Get probe data as features
        probe1 = df.iloc[:, 1].values  # Second column is probe 1
        
        # Scale features
        features_scaled = 1e7 * probe1
        
        # Add Gaussian noise if lambda > 0 (during training/data loading)
        if self.noise_lambda > 0:
            noise = np.random.normal(0, self.noise_lambda * np.std(features_scaled), features_scaled.shape)
            features_noisy = features_scaled + noise
        else:
            features_noisy = features_scaled

        # Reshape features - reshape to 2D tensor with shape (seq_len, 1)
        features = features_noisy.reshape(-1, 1)

        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        target[0] /= 250
        target[1] = (target[1] - 3000) / 3000
        # print(features.shape)
        # print(target)
        # # plt visualize the features
        # plt.plot(time, features)
        # plt.show()
        
        # Return features, target, and sequence length
        return features, target, len(features)

def collate_fn(batch):
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: x[2], reverse=True)
    
    # Separate features, targets, and lengths
    features, targets, lengths = zip(*batch)
    
    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True)
    
    # Convert to tensors
    targets = torch.stack(targets)
    lengths = torch.tensor(lengths)
    
    return features_padded, targets, lengths

class AudioSeqModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.3):
        super(AudioSeqModel, self).__init__()
        
        # 1D CNN for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size)
        )
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Tanh()
        )
        
        # Fully connected layers for prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Output: f and Z
        )
    
    def forward(self, x, lengths):
        # x shape: [batch_size, seq_len, 1]
        batch_size, seq_len, features = x.size()
        
        # Reshape for CNN: [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        
        # Apply CNN
        x = self.conv_layers(x)
        
        # Reshape back for LSTM: [batch, seq_len, hidden_size]
        x = x.permute(0, 2, 1)
        
        # Make sure lengths is on CPU
        lengths_cpu = lengths.cpu()
        
        # Pack padded sequence
        packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True)
        
        # Apply LSTM
        lstm_out, (hidden, _) = self.lstm(packed_x)
        
        # Unpack sequence
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        output = self.fc_layers(context_vector)
        
        return output

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training history
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        for features, targets, lengths in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(features, lengths)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets, lengths in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                outputs = model(features, lengths)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.show()
    
    return model

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for features, targets, lengths in test_loader:
            features = features.to(device)
            
            outputs = model(features, lengths)
            
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(targets.numpy())
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    # Calculate RMSE for f and Z
    rmse_f = np.sqrt(np.mean((predictions[:, 0] - true_values[:, 0]) ** 2))
    rmse_z = np.sqrt(np.mean((predictions[:, 1] - true_values[:, 1]) ** 2))
    
    print(f"RMSE for frequency (f): {rmse_f:.2f}")
    print(f"RMSE for impedance (Z): {rmse_z:.2f}")
    
    # Plot predictions vs true values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(true_values[:, 0], predictions[:, 0])
    plt.plot([min(true_values[:, 0]), max(true_values[:, 0])], 
             [min(true_values[:, 0]), max(true_values[:, 0])], 'r--')
    plt.xlabel('True f')
    plt.ylabel('Predicted f')
    plt.title('Frequency Prediction')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(true_values[:, 1], predictions[:, 1])
    plt.plot([min(true_values[:, 1]), max(true_values[:, 1])], 
             [min(true_values[:, 1]), max(true_values[:, 1])], 'r--')
    plt.xlabel('True Z')
    plt.ylabel('Predicted Z')
    plt.title('Impedance Prediction')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('prediction_performance.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Audio Sequence Model')
    parser.add_argument('--data_dir', type=str, default='outputs', help='Directory containing the CSV data')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--noise_lambda', type=float, default=0.0, help='Lambda coefficient for adding Gaussian noise for data augmentation')
    parser.add_argument('--load_checkpoint', type=str, default='audio_seq_model.pth', help='Path to load previous model checkpoint (optional)')
    parser.add_argument('--save_checkpoint', type=str, default='audio_seq_model.pth', help='Path to save the trained model checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data directory
    # data_dir = "outputs"
    
    # Create dataset
    dataset = AudioSequenceDataset(args.data_dir, noise_lambda=args.noise_lambda)
    
    # Split dataset
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.7)
    val_size = int(dataset_size * 0.15)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Create model
    model = AudioSeqModel(input_size=1, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout)
    model = model.to(device) # Move model to device early
    
    # load from previous checkpoint if specified and exists
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        print(f"Loading model checkpoint from {args.load_checkpoint}")
        model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
    
    # Train model
    model = train_model(model, train_loader, val_loader, num_epochs=args.epochs, learning_rate=args.lr)
    
    # Evaluate model
    evaluate_model(model, test_loader)
    
    # Save model
    torch.save(model.state_dict(), args.save_checkpoint)
    print(f"Model saved to {args.save_checkpoint}")
