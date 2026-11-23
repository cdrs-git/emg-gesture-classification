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

# class CNN(nn.Module):
#     def __init__(self, filters, output_dim, data_shape, dropout_rate=0.3):
#         super(CNN, self).__init__()
#         self.cnn = nn.Sequential(
#             # nn.Conv2d(1, filters, kernel_size=(data_shape[1], 3), padding=(int(data_shape[1]/2), 1)),
#             # nn.BatchNorm2d(filters),
#             # nn.ReLU(),
#             # nn.Dropout(dropout_rate),
#             # nn.AdaptiveAvgPool2d((1, 1))

#             nn.Conv2d(1, filters, kernel_size=(3, data_shape[2]), padding=(1, int(data_shape[2]/2))),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.AdaptiveAvgPool2d((1, 1))


#             # nn.Conv2d(1, filters, kernel_size=(3, 7), padding=(1, 3)),
#             # nn.BatchNorm2d(filters),
#             # nn.ReLU(),
#             # nn.Dropout(dropout_rate),
#             # nn.MaxPool2d((2, 2)),

#             # nn.Conv2d(filters, filters*2, kernel_size=(3, 3), padding=(1, 1)),
#             # nn.BatchNorm2d(filters*2),
#             # nn.ReLU(),
#             # nn.Dropout(dropout_rate),
#             # nn.MaxPool2d((2, 2)),

#             # nn.Conv2d(filters*2, filters*4, kernel_size=(3, 3), padding=(1, 1)),
#             # nn.BatchNorm2d(filters*4),
#             # nn.ReLU(),
#             # nn.Dropout(dropout_rate),
#             # nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.fc = nn.Linear(filters, output_dim)
        
#         # self.fc = nn.Linear(filters*4, output_dim)

#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)
    
# class CNN(nn.Module):
#     def __init__(self, filters, output_dim, data_shape, dropout_rate=0.3):
#         super(CNN, self).__init__()
#         self.cnn = nn.Sequential(
#             # Starting shape: [N, 1, C, T]

#             # # Layer 1
#             # # ---------------------------------------------------------------------------------------------------- #
#             # # Convolute over all channels
#             # nn.Conv2d(1, filters, kernel_size=(data_shape[1], 3), padding=(int((data_shape[1]/2)-1), 1)),
#             # # Normalise
#             # nn.BatchNorm2d(filters),
#             # # Activation function
#             # nn.ReLU(),
#             # # Dropout
#             # nn.Dropout(dropout_rate),
#             # # ---------------------------------------------------------------------------------------------------- #
#             # # Flatten to feed into final layer
#             # nn.Flatten(),
#             # # Classification layer              
#             # nn.Linear(filters * data_shape[2] * (data_shape[1]-1), output_dim)      


#             # Layer 1
#             # ---------------------------------------------------------------------------------------------------- #
#             # Convolute over all timesteps
#             nn.Conv2d(1, filters, kernel_size=(3, data_shape[2]), padding=(1, int((data_shape[2]/2)-1))),
#             # Normalise
#             nn.BatchNorm2d(filters),
#             # Activation function
#             nn.ReLU(),
#             # Dropout
#             nn.Dropout(dropout_rate),
#             # ---------------------------------------------------------------------------------------------------- #
#             # Flatten to feed into final layer
#             nn.Flatten(),
#             # Classification layer              
#             nn.Linear(filters * data_shape[1] * (data_shape[2]-1), output_dim) 

      
#         )

#     def forward(self, x):
#         return self.cnn(x)

# All cnn - pooling
# class CNN(nn.Module):
#     def __init__(self, filters, output_dim, data_shape, dropout_rate=0.3):
#         super(CNN, self).__init__()
#         # Channels (Electrodes), Timesteps 
#         C, T = data_shape[1], data_shape[2]

#         # Temporal Branch (goes across all channels, slides over time)
#         self.temporal = nn.Sequential(
#             nn.Conv2d(1, filters, kernel_size=(C, 3), padding=(0, 1)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(filters, filters, kernel_size=(1, 3), padding=(0, 1)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )

#         # Spatial Branch (goes across all time, slides over channels)
#         self.spatial = nn.Sequential(
#             nn.Conv2d(1, filters, kernel_size=(3, T), padding=(1, 0)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(filters, filters, kernel_size=(3, 1), padding=(1, 0)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )

#         # Fusion conv block
#         self.fusion = nn.Sequential(
#             nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)),  # compress spatially
#         )

#         self.classifier = nn.Linear(filters, output_dim)

#     def forward(self, x):
#         t = self.temporal(x)  # [N, F, 1, T]
#         s = self.spatial(x)   # [N, F, C, 1]

#         # Pad spatial and temporal to the same shape: [N, F, C, T]
#         t = t.expand(-1, -1, x.shape[2], -1)  # [N, F, C, T]
#         s = s.expand(-1, -1, -1, x.shape[3])  # [N, F, C, T]

#         fused = torch.cat([t, s], dim=1)   # [N, 2F, C, T]
#         fused = self.fusion(fused)         # [N, F, 1, 1]
#         fused = fused.view(x.size(0), -1)  # [N, F]
#         return self.classifier(fused)      # [N, output_dim]
    
# Residual function
# class ResidualFusion(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout_rate):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )
#         self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#         self.pool = nn.AdaptiveAvgPool2d((1, 1))

#     def forward(self, x):
#         identity = self.proj(x)
#         out = self.conv(x)
#         out = out + identity
#         out = F.relu(out)
#         out = self.pool(out)  # [N, out_channels, 1, 1]
#         return out
# class CNN(nn.Module):
#     def __init__(self, filters, output_dim, data_shape, dropout_rate=0.3):
#         super(CNN, self).__init__()
#         C, T = data_shape[1], data_shape[2]

#         self.temporal = nn.Sequential(
#             nn.Conv2d(1, filters, kernel_size=(C, 3), padding=(0, 1)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(filters, filters, kernel_size=(1, 3), padding=(0, 1)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )

#         self.spatial = nn.Sequential(
#             nn.Conv2d(1, filters, kernel_size=(3, T), padding=(1, 0)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(filters, filters, kernel_size=(3, 1), padding=(1, 0)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )

#         self.fusion = ResidualFusion(in_channels=filters * 2, out_channels=filters, dropout_rate=dropout_rate)
#         self.classifier = nn.Linear(filters, output_dim)

#     def forward(self, x):
#         t = self.temporal(x)                      # [N, F, 1, T]
#         s = self.spatial(x)                       # [N, F, C, 1]

#         t = t.expand(-1, -1, x.shape[2], -1)      # [N, F, C, T]
#         s = s.expand(-1, -1, -1, x.shape[3])      # [N, F, C, T]

#         fused = torch.cat([t, s], dim=1)          # [N, 2F, C, T]
#         fused = self.fusion(fused)                # [N, F, 1, 1]
#         fused = fused.view(x.size(0), -1)         # [N, F]
#         return self.classifier(fused)             # [N, output_dim]

# ALL CNN - no pooling
# class CNN(nn.Module):
#     def __init__(self, filters, output_dim, data_shape, dropout_rate=0.3):
#         super(CNN, self).__init__()
#         C, T = data_shape[1], data_shape[2]

#         # Temporal Branch (convolves across channels, slides over time)
#         self.temporal = nn.Sequential(
#             nn.Conv2d(1, filters, kernel_size=(C, 3), padding=(0, 1)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(filters, filters, kernel_size=(1, 3), padding=(0, 1)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )

#         # Spatial Branch (convolves across time, slides over channels)
#         self.spatial = nn.Sequential(
#             nn.Conv2d(1, filters, kernel_size=(3, T), padding=(1, 0)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(filters, filters, kernel_size=(3, 1), padding=(1, 0)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )

#         # Fusion block — no residual, no pooling
#         self.fusion = nn.Sequential(
#             nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),

#             nn.Conv2d(filters, filters, kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),

#             nn.Conv2d(filters, filters, kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters),
#             nn.ReLU()
#         )

#         # Final classifier — flatten full spatial map
#         self.classifier = nn.Sequential(
#             nn.Flatten(),  # shape: [N, filters * C * T]
#             nn.Linear(filters * C * T, output_dim)
#         )

#     def forward(self, x):
#         t = self.temporal(x)                     # [N, F, 1, T]
#         s = self.spatial(x)                      # [N, F, C, 1]

#         # Expand both to [N, F, C, T]
#         t = t.expand(-1, -1, x.shape[2], -1)
#         s = s.expand(-1, -1, -1, x.shape[3])

#         fused = torch.cat([t, s], dim=1)         # [N, 2F, C, T]
#         fused = self.fusion(fused)               # [N, F, C, T]

#         return self.classifier(fused)            # raw logits

# MLP fusion
class CNN(nn.Module):
    def __init__(self, filters, output_dim, data_shape, dropout_rate=0.3):
        super(CNN, self).__init__()
        # Channels (Electrodes), Timesteps 
        C, T = data_shape[1], data_shape[2]

        # Temporal Branch (goes across all channels, slides over time)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=(C, 3), padding=(0, 1)),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(filters, filters, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(filters, filters, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Spatial Branch (goes across all time, slides over channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(1, filters, kernel_size=(3, T), padding=(1, 0)),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(filters, filters, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(filters, filters, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Compute flatten sizes for the classifier
        temporal_out_dim = filters * T  # [N, filters, 1, T]
        spatial_out_dim = filters * C   # [N, filters, C, 1]
        fusion_dim = temporal_out_dim + spatial_out_dim

        # Classifier head after fusion
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim // 2, output_dim)
        )

    def forward(self, x):
        t = self.temporal(x)  # [N, filters, 1, T]
        s = self.spatial(x)   # [N, filters, C, 1]

        t = t.squeeze(2).flatten(1)  # [N, filters * T]
        s = s.squeeze(3).flatten(1)  # [N, filters * C]

        combined = torch.cat([t, s], dim=1)  # [N, fusion_dim]
        return self.classifier(combined)  # raw logits

# Serial - mlp head
# class CNN(nn.Module):
#     def __init__(self, filters, output_dim, data_shape, dropout_rate=0.3):
#         super(CNN, self).__init__()
#         C, T = data_shape[1], data_shape[2]  # Spatial height and width

#         self.feature_extractor = nn.Sequential(
#             # Block 1
#             nn.Conv2d(1, filters, kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.Conv2d(filters, filters, kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),  # Halve C and T

#             # Block 2
#             nn.Conv2d(filters, filters * 2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters * 2),
#             nn.ReLU(),
#             nn.Conv2d(filters * 2, filters * 2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(filters * 2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),  # Halve again

#             nn.Dropout(dropout_rate)
#         )

#         # Compute spatial size after 2x pooling (C//4, T//4)
#         reduced_c, reduced_t = C // 4, T // 4
#         flat_features = (filters * 2) * reduced_c * reduced_t

#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(flat_features, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(128, output_dim)
#         )

#     def forward(self, x):
#         x = self.feature_extractor(x)
#         return self.classifier(x)

# # Residual, MLP, parallel
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)

#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         # Shortcut projection if channels mismatch
#         self.shortcut = nn.Identity()
#         if in_channels != out_channels:
#             self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         residual = self.shortcut(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         out += residual
#         out = self.relu(out)
#         return out
# class CNN(nn.Module):
#     def __init__(self, filters, output_dim, data_shape, dropout_rate=0.3):
#         super(CNN, self).__init__()
#         C, T = data_shape[1], data_shape[2]

#         # Temporal Branch (Residual Block)
#         self.temporal = nn.Sequential(
#             nn.Conv2d(1, filters, kernel_size=(C, 3), padding=(0, 1)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             ResidualBlock(filters, filters, kernel_size=(1, 3), padding=(0, 1)),
#             nn.Dropout(dropout_rate)
#         )

#         # Spatial Branch (Residual Block)
#         self.spatial = nn.Sequential(
#             nn.Conv2d(1, filters, kernel_size=(3, T), padding=(1, 0)),
#             nn.BatchNorm2d(filters),
#             nn.ReLU(),
#             ResidualBlock(filters, filters, kernel_size=(3, 1), padding=(1, 0)),
#             nn.Dropout(dropout_rate)
#         )

#         # Fusion dimensions
#         temporal_out_dim = filters * T
#         spatial_out_dim = filters * C
#         fusion_dim = temporal_out_dim + spatial_out_dim

#         # Classifier MLP
#         self.classifier = nn.Sequential(
#             nn.Linear(fusion_dim, fusion_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(fusion_dim // 2, output_dim)
#         )

#     def forward(self, x):
#         t = self.temporal(x)  # [N, F, 1, T]
#         s = self.spatial(x)   # [N, F, C, 1]

#         t = t.squeeze(2).flatten(1)  # [N, filters * T]
#         s = s.squeeze(3).flatten(1)  # [N, filters * C]

#         combined = torch.cat([t, s], dim=1)  # [N, fusion_dim]
#         return self.classifier(combined)

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

    plots.plot_confusion_matrix(labels=all_labels, predictions=all_preds, database=database, method='CNN', test_num=test_number, display=False)

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

def train_test_cnn(database, test_number, data_train, labels_train, data_val, labels_val, data_test, labels_test):
    # Convert to tensors
    # Data: floats. Shape: [N, T, C]
    data_train = torch.tensor(data_train, dtype=torch.float32)
    data_val   = torch.tensor(data_val,   dtype=torch.float32)
    data_test  = torch.tensor(data_test,  dtype=torch.float32)
    
    # Labels: integer (long)
    labels_train = torch.tensor(labels_train, dtype=torch.long)
    labels_test  = torch.tensor(labels_test,  dtype=torch.long)
    labels_val   = torch.tensor(labels_val,   dtype=torch.long)

    # Transform into shape: [N, 1, C, T] for 2D CNN
    data_train = data_train.permute(0, 2, 1).unsqueeze(1)  
    data_val   = data_val.permute(0, 2, 1).unsqueeze(1)
    data_test  = data_test.permute(0, 2, 1).unsqueeze(1)

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
    
    # Define model hyperparameters
    output_dim = len(np.unique(labels_train))
    filters    = 256
    dropout    = 0.4
    epochs     = 64
    data_shape = data_train[0].shape
    print(f'Data shape = {data_shape}')

    # Create model
    model = CNN(filters=filters, output_dim=output_dim, dropout_rate=dropout, data_shape=data_shape)
    # Put model on GPU memory
    model = model.to(globals.DEVICE) 

    # Create training tools
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda")  # Helps speed up training

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)  # Adjust LR every 5 epochs

    # Reduce learning rate when the validation loss plateaus
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    

    # Train 
    training_time, train_acc, val_acc = train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=epochs, scaler=scaler)

    # Test
    test_acc, predict_time, predict_times_rate = test(model, test_loader, criterion, database=database, test_number=test_number)


    return train_acc, val_acc, test_acc, training_time, predict_time, predict_times_rate








