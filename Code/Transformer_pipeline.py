# Libraries
import torch
import time
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms # on-fly-augmentations

# My files
import globals
import data_collector
import plots
import filters

class AddGaussianNoise(object):
    """Custom transform to add Gaussian noise to a tensor."""
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Add Gaussian noise with specified mean and std
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

class TimeSeriesDataset(Dataset):
    def __init__(self, windows, labels, transform=None):
        self.windows   = windows
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        y = self.labels[idx]

        # On-the-fly augmentation
        if self.transform:
            x = self.transform(x)

        return x, y

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, num_layers, output_dim, max_seq_len=200, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        # Linear layer to project input channels to the Transformer's embedding dimension
        self.input_projection = nn.Linear(input_size, embed_dim)

        # Positional encoding to add sequence order information (learnable parameters)
        # Shape: [1, max_seq_len, embed_dim] to broadcast across batch
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Transformer encoder layer configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,              # Embedding dimension for attention
            nhead=num_heads,                # Number of attention heads
            dim_feedforward=embed_dim * 4,  # Feedforward network size (4x embed_dim)
            dropout=dropout,                # Dropout for regularization
            batch_first=True                # Input shape: [batch, seq_len, embed_dim]
        )

        # Stack multiple encoder layers to form the Transformer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head to map Transformer output to class logits
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),  # Reduce dimension for efficiency
            nn.ReLU(),                            # Non-linearity
            nn.Dropout(dropout),                  # Regularization
            nn.Linear(embed_dim // 2, output_dim)  # Output class logits
        )

        # Dropout layer applied before classification
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project input channels to embedding dimension
        # Input: [batch, seq_len, input_size] -> Output: [batch, seq_len, embed_dim]
        x = self.input_projection(x)

        # Add positional encoding to capture sequence order
        # Truncate positional encoding to match input sequence length
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]

        # Pass through Transformer encoder
        # Input: [batch, seq_len, embed_dim] -> Output: [batch, seq_len, embed_dim]
        x = self.transformer_encoder(x)

        # Pool the sequence by taking the mean across time steps
        # Input: [batch, seq_len, embed_dim] -> Output: [batch, embed_dim]
        x = x.mean(dim=1)

        # Apply dropout for regularization
        x = self.dropout(x)

        # Pass through classification head to get logits
        # Input: [batch, embed_dim] -> Output: [batch, output_dim]
        logits = self.classifier(x)

        return logits

def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)
 
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, epochs=20):
    print(f'Training started...')
    train_start = time.time()

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            # Use GPU
            x_batch = x_batch.to(globals.DEVICE)
            y_batch = y_batch.to(globals.DEVICE)

            # Reset gradients for batch
            optimizer.zero_grad()

            # Forward pass with AMP (Automatic Mixed Precision)
            with torch.amp.autocast("cuda"):
                output = model(x_batch)
                loss = criterion(output, y_batch)  

            # Backward pass and optimization with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate loss
            train_loss += loss.item()

            # Calculate accuracy
            preds = output.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            # Cyclic scheduler after each batch
            # scheduler.step()
        
        # Find average loss, average accuracy
        train_loss /= len(train_loader)
        train_acc = train_correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                # Use GPU
                x_val = x_val.to(globals.DEVICE)
                y_val = y_val.to(globals.DEVICE)

                # Calculate loss
                val_output = model(x_val)
                val_loss += criterion(val_output, y_val).item()

                # Calculate accuracy
                val_correct += (val_output.argmax(dim=1) == y_val).sum().item()
                total += y_val.size(0)

            # Calculate averages across batches
            val_loss /= len(val_loader)
            val_acc = val_correct / total

        # Cosine schedular updated every epoch
        scheduler.step()

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Epoch Time: {time.time()-start_time:.2f}s")
    
    training_time = time.time()-train_start
    print(f'Model trained. Training time: {training_time:.2f}s')

    return training_time, train_acc, val_acc

def test(model, test_loader, criterion, database, test_number):
    start_pred = time.time()
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_test, y_test in test_loader:
            # Use GPU
            x_test = x_test.to(globals.DEVICE)
            y_test = y_test.to(globals.DEVICE)

            # Calculate loss
            output = model(x_test)
            test_loss += criterion(output, y_test).item()

            # Calculate accuracy
            correct += (output.argmax(dim=1) == y_test).sum().item()
            total += y_test.size(0)

            # Collect all predictions and labels for confusion matrix
            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_test.cpu().numpy())

    # Calculate batch averages
    test_loss /= len(test_loader)
    acc = correct / total

    total_pred_time = time.time() - start_pred
    pred_rate = total_pred_time / total
    
    print(f'Model tested. Testing time: {total_pred_time:.2f}s')
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {acc*100:.2f}%")

    plots.plot_confusion_matrix(labels=all_labels, predictions=all_preds, database=database, method='Transformer', test_num=test_number, display=False)

    return acc, total_pred_time, pred_rate    

def pre_process(database, data, labels, repetitions):
    subdivide_start = time.time()

    # Split big windows into train, validation, and test sets
    data_train, labels_train, reps, data_val, labels_val, reps_val, data_test, labels_test, reps_test = data_collector.separate_repetitions_train_valid_test(
        database=database, 
        windows=data, 
        labels=labels, 
        rep_nums=repetitions, 
        random_sets=True, 
        proportions=[0.5, 0.2, 0.3]
        )
    
    # Gestures used for each set
    gesture_set_list = [reps, reps_val, reps_test]
    print(f'Datasets split')
    print(f"Train set: {len(labels_train)}, Val test: {len(labels_val)}, Test set: {len(labels_test)}")
    
    # Resample and subdivide into smaller, sliding windows
    original_fs     = globals.FS[database-1]
    target_fs       = 200
    window_width_ms = 200
    overlap_percent = 50
    
    # Split big windows into smaller, sliding windows
    data_train, labels_train = data_collector.resample_and_subdivide_np(database=database, windows=data_train, labels=labels_train, orig_fs=original_fs, target_fs=target_fs, window_width_ms=window_width_ms, overlap_percent=overlap_percent)
    data_test, labels_test   = data_collector.resample_and_subdivide_np(database=database, windows=data_test,  labels=labels_test,  orig_fs=original_fs, target_fs=target_fs, window_width_ms=window_width_ms, overlap_percent=overlap_percent)
    data_val, labels_val     = data_collector.resample_and_subdivide_np(database=database, windows=data_val,   labels=labels_val,   orig_fs=original_fs, target_fs=target_fs, window_width_ms=window_width_ms, overlap_percent=overlap_percent)

    print(f"Subdivision time: {time.time() - subdivide_start:.2f}s")
    print(f"Train set: {data_train.shape}, {labels_train.shape}. Val set: {data_val.shape}, {labels_val.shape}. Test set: {data_test.shape}, {labels_test.shape}")
    
    # plots.plot_window_before_after(emg_before=data[0], emg_after=data[1])

    # Balance datasets
    # Count of each class label in y
    class_counts_train = np.bincount(labels_train)
    class_counts_test  = np.bincount(labels_test)
    class_counts_val   = np.bincount(labels_val)

    # Get the index with the smallest count
    min_class_train = np.argmin(class_counts_train)
    min_class_test  = np.argmin(class_counts_test)
    min_class_val   = np.argmin(class_counts_val)

    # Get the smallest count
    min_count_train = class_counts_train[min_class_train]
    min_count_test  = class_counts_test[min_class_test]
    min_count_val   = class_counts_val[min_class_val]

    print(f"Smallest train class data: {min_class_train}, Count: {min_count_train}")
    print(f"Smallest test class data:  {min_class_test}, Count: {min_count_test}")
    print(f"Smallest val class data:   {min_class_val}, Count: {min_count_val}")

    # Balance classes
    target_per_class = min_count_train
    data_train, labels_train = data_collector.balance_data(x=data_train, y=labels_train, target_per_class=target_per_class, class_counts=class_counts_train)
    target_per_class = min_count_val
    data_val, labels_val     = data_collector.balance_data(x=data_val,  y=labels_val,    target_per_class=target_per_class, class_counts=class_counts_val)
    target_per_class = min_count_test
    data_test, labels_test   = data_collector.balance_data(x=data_test, y=labels_test,   target_per_class=target_per_class, class_counts=class_counts_test)
    
    print(f"Balanced sets: Tr{data_train.shape}, Va{data_val.shape}, Te{data_test.shape}")
    plots.plot_class_distribution(train_labels=labels_train, val_labels=labels_val, test_labels=labels_test, database=database, display=False)

    # Scale all datasets per channel (fit on training only)
    scale_start = time.time()
    data_train, data_val, data_test, scalars = filters.scale_per_channel(data_train, data_val, data_test)

    # plots.plot_all_windows(data=data, labels=labels)

    scale_time = time.time() - scale_start
    scale_rate = scale_time / (data_train.shape[0]+data_test.shape[0]+data_val.shape[0])
    print(f"Scale time: {scale_time:.2f}s, Scale rate: {scale_rate:.2f}s per input")
    print(f"Scaled sets: Tr{data_train.shape}, Va{data_val.shape}, Te{data_test.shape}")

    return data_train, labels_train, data_val, labels_val, data_test, labels_test, gesture_set_list

def train_test_transformer(database, test_number, data_train, labels_train, data_val, labels_val, data_test, labels_test):
    # Convert to tensors
    # Data: floats. Shape: [N, T, C]
    data_train = torch.tensor(data_train, dtype=torch.float32)
    data_val   = torch.tensor(data_val,   dtype=torch.float32)
    data_test  = torch.tensor(data_test,  dtype=torch.float32)
    
    # Labels: integer (long)
    labels_train = torch.tensor(labels_train, dtype=torch.long)
    labels_test  = torch.tensor(labels_test,  dtype=torch.long)
    labels_val   = torch.tensor(labels_val,   dtype=torch.long)

    # RNN wants shape [N, T, C] - no transform

    # Define the augmentation pipeline for training
    train_transform = transforms.Compose([
        AddGaussianNoise(mean=0.0, std=0.1),
    ])

    # Convert to datasets
        # Only transform train set
    train_ds = TimeSeriesDataset(data_train, labels_train, transform=train_transform) 
    val_ds   = TimeSeriesDataset(data_val,   labels_val,   transform=None)
    test_ds  = TimeSeriesDataset(data_test,  labels_test,  transform=None)

    # Convert datasets to dataloaders
    batch_size   = 1024
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=6, persistent_workers=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,               num_workers=6, persistent_workers=True, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,               num_workers=6, persistent_workers=True, pin_memory=True)

    # Debug batch
    for batch in train_loader:
        x_batch, y_batch = batch  # Get batch of data
        print(f"Data batch shape: {x_batch.shape}, Label batch shape: {y_batch.shape}")
        break  # Stop after printing one batch
    
    data_shape = data_train[0].shape
    print(f'Data shape = {data_shape}')

    # Define model hyperparameters
    input_size = data_shape[1]   # Number of channels (C)
    max_seq_len = data_shape[0]  # Sequence length (T)
    output_dim = len(np.unique(labels_train)) 
    embed_dim = 256  # Embedding dimension for Transformer
    num_heads = 8    # Number of attention heads
    num_layers = 4   # Number of Transformer layers
    dropout = 0.1    
    epochs = 64   
    

    # Create model
    model = TransformerClassifier(
        input_size=input_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        output_dim=output_dim,
        max_seq_len=max_seq_len,
        dropout=dropout
    )
    model = model.to(globals.DEVICE) 

    # Create training tools
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5) 
    scaler = torch.amp.GradScaler("cuda")  # Helps speed up training

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # Adjust LR every 5 epochs
    
    # Train 
    training_time, train_acc, val_acc = train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=epochs, scaler=scaler)

    # Test
    test_acc, predict_time, predict_times_rate = test(model, test_loader, criterion, database=database, test_number=test_number)


    return train_acc, val_acc, test_acc, training_time, predict_time, predict_times_rate








