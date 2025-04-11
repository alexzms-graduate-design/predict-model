import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import glob
import re
import math # Add math import
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# --- Data Loading (Mostly unchanged from modelling_seq.py) ---
class AudioSequenceDataset(Dataset):
    def __init__(self, data_dir, max_seq_length=6000):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length # Store max_seq_length
        self.file_paths = []
        self.targets = []

        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

        for file_path in csv_files:
            filename = os.path.basename(file_path)
            match = re.search(r"output_f=(\d+)_Z=(\d+)_T=(\d+\.\d+)\.csv", filename)
            if match:
                f = float(match.group(1))
                Z = float(match.group(2))

                # Read CSV file, skip header rows
                df = pd.read_csv(file_path, skiprows=4)

                # Get probe data as features
                probe1 = df.iloc[:, 1].values  # Second column is probe 1

                # Combine features - reshape to 2D tensor with shape (seq_len, 1)
                features = (1e7*probe1).reshape(-1, 1)

                # Truncate or pad features (Important for PositionalEncoding)
                if len(features) > self.max_seq_length:
                    features = features[:self.max_seq_length]
                # Padding is handled in collate_fn

                # Convert to tensor
                features = torch.tensor(features, dtype=torch.float32)
                target = torch.tensor((f, Z), dtype=torch.float32)
                target[0] /= 250
                target[1] /= 6000

                self.file_paths.append(file_path) # Keep track for __len__
                self.targets.append((features, target, len(features))) # Store pre-processed data

    def __len__(self):
        return len(self.targets) # Use length of processed targets

    def __getitem__(self, idx):
        # Return pre-processed features, target, and length
        return self.targets[idx]

# --- Collate Function (Modified for Transformer padding mask) ---
def collate_fn(batch):
    # No need to sort by length for Transformer with padding mask
    # batch.sort(key=lambda x: x[2], reverse=True)

    # Separate features, targets, and lengths
    features, targets, lengths = zip(*batch)

    # Pad sequences
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0) # Pad with 0

    # Create padding mask (True where padded, False otherwise)
    max_len = features_padded.size(1)
    # Mask shape: [batch_size, seq_len]
    padding_mask = torch.arange(max_len)[None, :] >= torch.tensor(lengths)[:, None]

    # Convert targets to tensor
    targets = torch.stack(targets)

    return features_padded, targets, padding_mask # Return mask instead of lengths

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model) # Shape [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: [1, max_len, d_model] for batch compatibility
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding up to the sequence length of x
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- Transformer Model ---
class AudioTransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=128, nhead=8, num_encoder_layers=6, dim_feedforward=512, dropout=0.1, max_len=6000):
        super().__init__()
        self.d_model = d_model
        self.input_embed = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        # Output layer: takes average of sequence output
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2) # Output: f and Z
        )

    def forward(self, src, src_key_padding_mask=None):
        # src shape: [batch_size, seq_len, input_size]
        # src_key_padding_mask shape: [batch_size, seq_len]

        # Embed input
        src = self.input_embed(src) * math.sqrt(self.d_model) # Shape: [batch, seq_len, d_model]
        src = self.pos_encoder(src)

        # Apply Transformer Encoder
        # The mask should be True for positions to be ignored
        memory = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        # memory shape: [batch, seq_len, d_model]

        # Aggregate output: Average pooling over non-padded sequence elements
        if src_key_padding_mask is not None:
             # Invert mask: True for valid elements, False for padding
            mask_inv = ~src_key_padding_mask.unsqueeze(-1) # Shape [batch, seq_len, 1]
            # Sum valid elements and divide by number of valid elements
            output = (memory * mask_inv).sum(dim=1) / mask_inv.sum(dim=1).clamp(min=1e-8) # Avoid division by zero
        else:
            # If no mask, just average over sequence length
            output = memory.mean(dim=1) # Shape: [batch, d_model]


        # Final prediction
        output = self.output_layer(output) # Shape: [batch, 2]
        return output

# --- Training Function (Modified for Transformer mask) ---
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # Loop unpacks features, targets, mask
        for features, targets, padding_mask in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            padding_mask = padding_mask.to(device) # Move mask to device

            # Pass mask to model's forward method
            outputs = model(features, src_key_padding_mask=padding_mask)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping can help stabilize Transformer training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            # Loop unpacks features, targets, mask
            for features, targets, padding_mask in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                padding_mask = padding_mask.to(device)

                # Pass mask to model's forward method
                outputs = model(features, src_key_padding_mask=padding_mask)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Transformer Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('transformer_training_history.png') # Save with new name
    plt.show()

    return model

# --- Evaluation Function (Modified for Transformer mask) ---
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    predictions = []
    true_values = []

    with torch.no_grad():
        # Loop unpacks features, targets, mask
        for features, targets, padding_mask in test_loader:
            features = features.to(device)
            padding_mask = padding_mask.to(device) # Move mask to device

            # Pass mask to model's forward method
            outputs = model(features, src_key_padding_mask=padding_mask)

            predictions.extend(outputs.cpu().numpy())
            true_values.extend(targets.numpy()) # Targets are already on CPU via DataLoader

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
    plt.scatter(true_values[:, 0], predictions[:, 0], alpha=0.5)
    plt.plot([min(true_values[:, 0]), max(true_values[:, 0])],
             [min(true_values[:, 0]), max(true_values[:, 0])], 'r--', label='Ideal')
    plt.xlabel('True f')
    plt.ylabel('Predicted f')
    plt.title('Transformer Frequency Prediction')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(true_values[:, 1], predictions[:, 1], alpha=0.5)
    plt.plot([min(true_values[:, 1]), max(true_values[:, 1])],
             [min(true_values[:, 1]), max(true_values[:, 1])], 'r--', label='Ideal')
    plt.xlabel('True Z')
    plt.ylabel('Predicted Z')
    plt.title('Transformer Impedance Prediction')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('transformer_prediction_performance.png') # Save with new name
    plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    # Hyperparameters & Configuration
    DATA_DIR = "outputs"
    MAX_SEQ_LENGTH = 6000 # Define explicitly
    BATCH_SIZE = 48 # Adjust based on GPU memory
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001 # Transformers often benefit from smaller LR
    NUM_WORKERS = 2 # Set based on your system

    # Model Hyperparameters
    D_MODEL = 128
    NHEAD = 8 # Must divide D_MODEL
    NUM_ENCODER_LAYERS = 4 # Adjust based on complexity/data
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset
    print("Loading dataset...")
    dataset = AudioSequenceDataset(DATA_DIR, max_seq_length=MAX_SEQ_LENGTH)
    print(f"Dataset size: {len(dataset)}")

    # Split dataset
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError(f"No valid data found in directory: {DATA_DIR}")

    train_size = int(dataset_size * 0.7)
    val_size = int(dataset_size * 0.15)
    # Ensure test_size is at least 1 if dataset is small
    test_size = max(1, dataset_size - train_size - val_size)
    # Adjust train/val if necessary due to rounding or small dataset
    val_size = dataset_size - train_size - test_size
    if train_size <=0 or val_size <=0 or test_size <=0:
         raise ValueError(f"Dataset split resulted in non-positive sizes: Train={train_size}, Val={val_size}, Test={test_size}")


    print(f"Splitting dataset: Train={train_size}, Val={val_size}, Test={test_size}")
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # for reproducibility
    )

    # Create data loaders (using modified collate_fn)
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True # Helps speed up CPU->GPU transfer if using CUDA
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Create Transformer model
    print("Creating model...")
    model = AudioTransformerModel(
        input_size=1,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LENGTH # Pass max_len for positional encoding
    )

    # Count model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")


    # Train model
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

    # Evaluate model
    print("Starting evaluation...")
    evaluate_model(model, test_loader)

    # Save model
    model_save_path = 'audio_transformer_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}") 