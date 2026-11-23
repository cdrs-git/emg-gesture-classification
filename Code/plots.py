import matplotlib
matplotlib.use('Agg') # Comment to enable displaying again
import matplotlib.pyplot as plt

from collections import Counter
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.signal import resample

import globals
import mat_file
import filters

def plot_class_distribution(train_labels, val_labels, test_labels, database, display=False):
    # Convert tensors to numpy arrays (if necessary)
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.numpy()
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.numpy()

    # Count occurrences of each class in training and testing sets
    train_counts = Counter(train_labels)
    test_counts  = Counter(test_labels)
    val_counts   = Counter(val_labels)

    # Get all unique classes
    classes = sorted(set(train_counts.keys()).union(set(val_counts.keys())).union(set(test_counts.keys())))

    # Get class counts for each set, filling in missing classes with 0
    train_values = [train_counts.get(cls, 0) for cls in classes]
    val_values   = [val_counts.get(cls, 0) for cls in classes]
    test_values  = [test_counts.get(cls, 0) for cls in classes]

    # Create side-by-side bar plots
    x = np.arange(len(classes))  # X-axis positions
    width = 0.25  # Width of bars

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, train_values, width=width, color='skyblue', label='Train')
    plt.bar(x, val_values, width=width, color='lightgreen', label='Validation')
    plt.bar(x + width, test_values, width=width, color='salmon', label='Test')

    plt.xlabel('Class')
    plt.ylabel('Count')
    # plt.title('Class Distribution in Train, Validation, and Test Sets')
    plt.xticks(ticks=x, labels=classes, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(f'Code/Plots/DB{database}_class_distr.png', dpi=300, bbox_inches='tight')
    if display:
        plt.show()
    else:
        plt.close()

def plot_confusion_matrix(labels, predictions, database, method, display=False, test_num=-1):
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Reverse the y-axis and move x-axis to the top
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                     xticklabels=np.unique(labels), 
                     yticklabels=np.unique(labels))
    
    ax.xaxis.set_ticks_position('top')  # Move x-axis to the top
    ax.xaxis.set_label_position('top')  # Move x-axis label to the top
    # plt.gca().invert_yaxis()  # Reverse y-axis so it starts from the top
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    # plt.title('Confusion Matrix')

    plt.tight_layout()  # Prevent clipping
    plt.savefig(f'Code/Plots/{method}/DB{database}_{method}_conf_matrix_{test_num+1}.png', dpi=300, bbox_inches='tight')

    if display:
        plt.show()
    else:
        plt.close()

def plot_emg_fft_from_file(database, subject=1, exercise=1):
    if database == 1:
        file_str = f'Ninapro_data/DB{database}/S{subject}_A1_E{exercise}.mat'
    else:
        file_str = f'Ninapro_data/DB{database}/S{subject}_E{exercise}_A1.mat'

    data_path = globals.PROJECT_DIR / file_str
    _, _, emg, *_ = mat_file.extract_mat_file(data_path)

    freqs, fft_magnitudes = filters.compute_multichannel_fft(data=emg, sampling_rate=globals.FS[database-1], apply_window=True)

    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # Subplot 1: Raw EMG heatmap
    im = axs[0].imshow(emg.T, aspect='auto', cmap='seismic', interpolation='nearest')
    fig.colorbar(im, ax=axs[0], label='Amplitude')
    axs[0].set_title(f'Raw EMG - DB{database}, Subject {subject}, Exercise {exercise}')
    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('Channel')

    # Subplot 2: FFT frequency spectrum
    n_channels = emg.shape[1]
    for ch in range(n_channels):
        axs[1].plot(freqs, fft_magnitudes[:, ch], label=f'Channel {ch+1}')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Frequency Spectrum of Multi-Channel EMG')
    axs[1].legend(loc='upper right', ncol=2)
    axs[1].grid(True)
    axs[1].set_xlim(0, globals.FS[database-1]/2)  # Limit to Nyquist frequency

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(f'Code/Plots/FFT/DB{database}_S{subject}_E{exercise}_FFT.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_raw_data(data_path):
    # Extract .mat variables
    subject, exercise, emg, stimulus, restimulus, repetition, rerepetition = mat_file.extract_mat_file(data_path)

    print('Subject: ',  subject)
    print('Exercise: ', exercise)

    smallest = min(stimulus) 
    print(f'Stimulus range: {min(x for x in stimulus if x > smallest)} to {max(stimulus)}')

    smallest = min(repetition) 
    print(f'Repetition range: {min(x for x in repetition if x > smallest)} to {max(repetition)}')

    # avg, min_d, max_d = measure_positive_durations(stimulus)
    # print(f"Average Duration: {avg}, Min Duration: {min_d}, Max Duration: {max_d}")
    # avg, min_d, max_d = measure_positive_durations(restimulus)
    # print(f"Average Duration: {avg}, Min Duration: {min_d}, Max Duration: {max_d}")
    # avg, min_d, max_d = measure_positive_durations(repetition)
    # print(f"Average Duration: {avg}, Min Duration: {min_d}, Max Duration: {max_d}")
    # avg, min_d, max_d = measure_positive_durations(rerepetition)
    # print(f"Average Duration: {avg}, Min Duration: {min_d}, Max Duration: {max_d}")

    # Create the figure and axes
    fig, axs = plt.subplots(5, 1, figsize=(10, 8))
    im = axs[0].imshow(emg.T, aspect='auto', cmap='seismic', interpolation='nearest')  # Transpose to have 12 rows
    fig.colorbar(im, ax=axs[0], label='Value')  # Correct way to add colorbar in subplots
    axs[0].set_title('EMG')

    axs[1].plot(stimulus)
    axs[1].set_title('Stimulus')
    axs[2].plot(restimulus)
    axs[2].set_title('Restimulus')
    axs[3].plot(repetition)
    axs[3].set_title('Repetition')
    axs[4].plot(rerepetition)
    axs[4].set_title('Rerepetition')
    # for x in range(5):
        # axs[x].set_xlim(10000, 12000)
    plt.tight_layout()
    plt.show()

def plot_all_raw_data(database):
    for i in range(globals.SUBJECTS[database-1]):
        for j in range(globals.EXERCISES[database-1]):
            if database == 1:
                file_str = f'Ninapro_data/DB{database}/S{i+1}_A1_E{j+1}.mat'
            else:
                file_str = f'Ninapro_data/DB{database}/S{i+1}_E{j+1}_A1.mat'

            data_path = globals.PROJECT_DIR / file_str
            plot_raw_data(data_path=data_path)

def plot_windows_stats(windows, labels, stats):
    """
    Plots a 3x3 grid of windows based on min, avg, and max statistics.
    """
    indices = {
        "max": stats["max"]["index"],
        "avg": stats["avg"]["index"],
        "min": stats["min"]["index"]
    }
    
    plt.figure(figsize=(10, 10))
    positions = [(0, "max"), (1, "avg"), (2, "min")]  # Row mapping
    
    for row, stat in positions:
        idx = indices[stat]
        for col, offset in enumerate([-1, 0, 1]):
            adj_idx = idx + offset
            if 0 <= adj_idx < len(windows):
                plt.subplot(3, 3, row * 3 + col + 1)
                plt.imshow(windows[adj_idx].T, aspect='auto', cmap='seismic', interpolation='nearest')
                plt.colorbar(label='Value')
                plt.title(f'Class = {labels[adj_idx]}')
    
    plt.tight_layout()
    plt.show()

def plot_all_windows(data, labels, grid_size=(3, 3)):
    # for window in range(len(windows)):
    #     plt.figure(figsize=(18, 6))  # Adjust figure size
    #     plt.imshow(windows[window].T, aspect='auto', cmap='seismic', interpolation='nearest')  # Transpose to have rows for each sensors
    #     plt.colorbar(label='Value')
    #     plt.xlabel('Samples')
    #     plt.ylabel('Features (Columns)')
    #     plt.title('Class =' + str(labels[window]))
    #     plt.show()

    """
    Plots images in a grid format and iterates through them in batches.
    Closes the figure before displaying the next batch.
    
    :param windows: List or array of images (assumed to be NumPy arrays)
    :param labels: List of corresponding labels for the images
    :param grid_size: Tuple indicating the grid size (rows, cols)
    """

    # Convert labels to a list of strings (if not already)
    labels = list(map(str, labels))

    # Sort labels and reorder windows accordingly
    sorted_pairs = sorted(zip(labels, data), key=lambda x: x[0])

    # Unzip the sorted pairs
    sorted_labels, sorted_data = zip(*sorted_pairs)

    # Convert back to lists
    labels = list(sorted_labels)
    data = list(sorted_data) 

    # Plotting
    num_images = len(data)
    batch_size = grid_size[0] * grid_size[1]
    index = 0
    
    while index < num_images:
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1] * 2, grid_size[0] * 2))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for i in range(batch_size):
            if index >= num_images:
                break
            
            ax = axes[i]
            ax.imshow(data[index].T, aspect='auto', cmap='seismic', interpolation='nearest')
            ax.set_title(str(labels[index]))
            index += 1
            
        # Hide any unused subplots
        for j in range(i + 1, batch_size):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()

def plot_window_size_stats(windows, stats):
    """
    Plots a bar graph of the window sizes and highlights the max, min, and average sizes.
    
    Parameters:
        windows (list of lists): A list where each element is a list (window) of varying length.
        stats (dict): A dictionary containing max, min, and average window sizes with their indices.
    """
    if not windows:
        print("No data to plot.")
        return
    
    window_sizes = [len(window) for window in windows]
    
    # Plot the bar graph
    plt.figure(figsize=(4, 4))
    plt.bar(range(len(window_sizes)), window_sizes, color='skyblue')
    
    # Mark max, min, and average sizes
    plt.axhline(stats['max']['size'], color='red', linestyle='--', label=f"Max size: {stats['max']['size']} (Index: {stats['max']['index']})")
    plt.axhline(stats['min']['size'], color='green', linestyle='--', label=f"Min size: {stats['min']['size']} (Index: {stats['min']['index']})")
    plt.axhline(stats['avg']['size'], color='orange', linestyle='--', label=f"Average size: {stats['avg']['size']} (Index: {stats['avg']['index']})")
    
    plt.xlabel('Window Index')
    plt.ylabel('Window Size')
    plt.title(f'Window Sizes Distribution. Total = {len(windows)}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_training_graphs(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs):
    # Plot training loss and accuracy
    plt.figure(figsize=(12, 6))

    # Subplot for Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', color='b', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, marker='o', color='g', linestyle='dashed', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # Subplot for Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', color='b', label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, marker='o', color='g', linestyle='dashed', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_data_and_mask(data, mask):
    """
    Plots a 660x10 image as a heatmap and a 1D mask of width 660 below it.

    Parameters:
        image (numpy.ndarray): A 660x10 array representing the image.
        mask (numpy.ndarray): A 1D array of length 660 representing the mask.
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 3), gridspec_kw={'height_ratios': [10, 1]})
    
    # Plot the heatmap
    axes[0].imshow(data.T, aspect='auto', cmap='seismic', interpolation='nearest')
    
    # Plot the mask as a binary colormap
    axes[1].imshow(mask[np.newaxis, :], cmap='gray', aspect='auto')
    
    plt.tight_layout()
    plt.show()

# def plot_resampling_comparison(data_path, database, start_idx=10000, end_idx=12000):
    # """
    # Plot EMG, stimulus, and restimulus before and after resampling to 200 Hz.
    # Args:
    #     data_path (str or Path): Path to .mat file.
    #     database (int): NinaPro database number (1-5, 7).
    #     start_idx (int): Start index for zoomed-in view (default: 10000).
    #     end_idx (int): End index for zoomed-in view (default: 12000).
    # """
    # # Extract .mat variables
    # subject, exercise, emg, stimulus, restimulus, repetition, rerepetition = mat_file.extract_mat_file(data_path)

    # # Original sampling frequency
    # orig_fs = globals.FS[database - 1]
    # target_fs = 200.0

    # # Before resampling (raw data)
    # emg_before = emg
    # restimulus_before = restimulus
    # stimulus_before = stimulus

    # # Resample to 200 Hz
    # if orig_fs != target_fs:
    #     num_samples = int(round(emg.shape[0] * target_fs / orig_fs))
    #     emg_after = resample(emg, num_samples, axis=0)
    #     # Resample restimulus with nearest-neighbor interpolation
    #     orig_time = np.arange(restimulus.size) * (1 / orig_fs)
    #     new_time = np.arange(num_samples) * (1 / target_fs)
    #     restimulus_after = np.interp(new_time, orig_time, restimulus, left=0, right=0)
    #     restimulus_after = np.round(restimulus_after).astype(int)
    #     stimulus_after = np.interp(new_time, orig_time, stimulus, left=0, right=0)  # For consistency
    #     stimulus_after = np.round(stimulus_after).astype(int)
    # else:
    #     emg_after = emg
    #     restimulus_after = restimulus
    #     stimulus_after = stimulus

    # # Print metadata
    # print(f'Subject: {subject}, Exercise: {exercise}')
    # print(f'Original FS: {orig_fs} Hz, Resampled FS: {target_fs} Hz')
    # print(f'EMG shape before: {emg_before.shape}, after: {emg_after.shape}')
    # print(f'Restimulus unique before: {np.unique(restimulus_before)}')
    # print(f'Restimulus unique after: {np.unique(restimulus_after)}')

    # # Create figure with subplots (2 columns: before, after)
    # fig, axs = plt.subplots(3, 2, figsize=(15, 8), sharex='col')

    # # EMG Heatmaps
    # im_before = axs[0, 0].imshow(emg_before.T, aspect='auto', cmap='seismic', interpolation='nearest')
    # fig.colorbar(im_before, ax=axs[0, 0], label='Value')
    # axs[0, 0].set_title(f'EMG Before ({orig_fs} Hz)')

    # im_after = axs[0, 1].imshow(emg_after.T, aspect='auto', cmap='seismic', interpolation='nearest')
    # fig.colorbar(im_after, ax=axs[0, 1], label='Value')
    # axs[0, 1].set_title(f'EMG After ({target_fs} Hz)')

    # # Stimulus Plots
    # axs[1, 0].plot(stimulus_before)
    # axs[1, 0].set_title('Stimulus Before')
    # axs[1, 1].plot(stimulus_after)
    # axs[1, 1].set_title('Stimulus After')

    # # Restimulus Plots
    # axs[2, 0].plot(restimulus_before)
    # axs[2, 0].set_title('Restimulus Before')
    # axs[2, 1].plot(restimulus_after)
    # axs[2, 1].set_title('Restimulus After')

    # # Adjust layout
    # plt.tight_layout()
    # plt.show()

def plot_resampling_comparison(data_path, database, start_idx=20000, end_idx=24000):
    subject, exercise, emg, stimulus, restimulus, _, _ = mat_file.extract_mat_file(data_path)
    orig_fs = globals.FS[database - 1]
    target_fs = 200.0

    # Before: Raw
    emg_before = emg
    restimulus_before = restimulus

    # Band-pass then resample
    lowcut = 20
    highcut = min(400, orig_fs / 2 - 1e-6)
    emg_filtered = filters.bandpass_filter(emg, lowcut, highcut, fs=orig_fs)
    if orig_fs != target_fs:
        num_samples = int(round(emg_filtered.shape[0] * target_fs / orig_fs))
        emg_after = resample(emg_filtered, num_samples, axis=0)
        orig_time = np.arange(restimulus.size) * (1 / orig_fs)
        new_time = np.arange(num_samples) * (1 / target_fs)
        restimulus_after = np.interp(new_time, orig_time, restimulus, left=0, right=0)
        restimulus_after = np.round(restimulus_after).astype(int)
    else:
        emg_after = emg_filtered
        restimulus_after = restimulus

    print(f"EMG before range: {np.min(emg_before):.4f} to {np.max(emg_before):.4f}")
    print(f"EMG after range: {np.min(emg_after):.4f} to {np.max(emg_after):.4f}")

    fig, axs = plt.subplots(2, 2, figsize=(15, 6), sharex='col')
    vmin, vmax = min(np.min(emg_before), np.min(emg_after)), max(np.max(emg_before), np.max(emg_after))
    axs[0, 0].imshow(emg_before.T, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title(f'EMG Raw ({orig_fs} Hz)')
    axs[0, 1].imshow(emg_after.T, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title(f'EMG Filtered+Resampled ({target_fs} Hz)')
    axs[1, 0].plot(restimulus_before)
    axs[1, 0].set_title('Restimulus Before')
    axs[1, 1].plot(restimulus_after)
    axs[1, 1].set_title('Restimulus After')

    
    plt.tight_layout()
    plt.show()

def plot_window_before_after(emg_before, emg_after):
    # fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharex='col')
    # axs[0].imshow(emg_before.T, aspect='auto', cmap='seismic')
    # axs[0].set_title(f'Before')
    # axs[0].colorbar(label='Value')
    # axs[1].imshow(emg_after.T, aspect='auto', cmap='seismic')
    # axs[1].set_title(f'After')
    # axs[1].colorbar(label='Value')

    # plt.tight_layout()
    # plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharex='col')

    # Plot the 'before' image and add colorbar
    im1 = axs[0].imshow(emg_before.T, aspect='auto', cmap='seismic')
    axs[0].set_title(f'Before')
    cbar1 = fig.colorbar(im1, ax=axs[0], orientation='vertical')
    cbar1.set_label('Value')

    # Plot the 'after' image and add colorbar
    im2 = axs[1].imshow(emg_after.T, aspect='auto', cmap='seismic')
    axs[1].set_title(f'After')
    cbar2 = fig.colorbar(im2, ax=axs[1], orientation='vertical')
    cbar2.set_label('Value')

    plt.tight_layout()
    plt.show()

def plot_rf_feature_importances(importances, database, feature_list, test_num=-1):
    
    num_electrodes = globals.SENSORS[database-1]
    feature_aspects = int(importances.shape[0]/num_electrodes)
    importances_per_electrode = importances.reshape(num_electrodes, feature_aspects)  # num_electrodes × 11 features

    sns.heatmap(importances_per_electrode, annot=False, cmap="viridis",
            xticklabels=feature_list,
            yticklabels=[f"{i+1}" for i in range(globals.SENSORS[database-1])])
    plt.xlabel("Feature Type")
    plt.ylabel("Electrode Channel")
    plt.tight_layout()
    plt.savefig(f'Code/Plots/RF/database_{database}_random_forest_feature_importance_{test_num+1}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_per_electrode(features, labels, enabled_features, feature_name, num_electrodes):
    assert feature_name in enabled_features, f"{feature_name} not in enabled_features"

    feature_index = enabled_features.index(feature_name)
    classes = np.unique(labels)
    num_classes = len(classes)

    # Plot settings
    ncols = 4  # Customize: how many plots per row
    nrows = int(np.ceil(num_electrodes / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for e in range(num_electrodes):
        ax = axs[e // ncols][e % ncols]
        col_index = e * len(enabled_features) + feature_index

        data_per_class = [features[labels == c, col_index] for c in classes]

        ax.boxplot(data_per_class, labels=classes, showfliers=False)
        ax.set_title(f"{feature_name.upper()} - Electrode {e}")
        ax.set_xlabel("Class")
        ax.set_ylabel(feature_name.upper())
        ax.grid(True)

    # Hide any unused subplots
    for i in range(num_electrodes, nrows * ncols):
        fig.delaxes(axs[i // ncols][i % ncols])

    fig.tight_layout()
    plt.show()

def plot_emg_comparison(emg1, emg2, xlim=None):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    # Subplot 1: EMG line plot
    for ch in range(emg1.shape[1]):
        axs[0].plot(emg1[:, ch], label=f'Channel {ch + 1}')

    axs[0].set_title(f'EMG1')
    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend(loc='upper right', fontsize='small')

    # Subplot 2: EMG line plot 2
    for ch in range(emg2.shape[1]):
        axs[1].plot(emg2[:, ch], label=f'Channel {ch + 1}')

    axs[1].set_title(f'EMG2')
    axs[1].set_xlabel('Sample')
    axs[1].set_ylabel('Amplitude')
    axs[1].legend(loc='upper right', fontsize='small')

    if xlim:
        axs[0].set_xlim(xlim[0], xlim[1])
        axs[1].set_xlim(xlim[0], xlim[1])

    plt.tight_layout()
    plt.show()

def plot_fft_comparison(channels, fs, freqs, freqs2, fft_magnitudes, fft_magnitudes2):
    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # Subplot 2: FFT frequency spectrum
    n_channels = channels
    for ch in range(n_channels):
        axs[0].plot(freqs, fft_magnitudes[:, ch], label=f'Channel {ch+1}')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Frequency Spectrum of Multi-Channel EMG')
    axs[0].legend(loc='upper right', ncol=2)
    axs[0].grid(True)
    axs[0].set_xlim(0, fs/2)  # Limit to Nyquist frequency

    # Subplot 2: FFT frequency spectrum
    for ch in range(n_channels):
        axs[1].plot(freqs2, fft_magnitudes2[:, ch], label=f'Channel {ch+1}')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Frequency Spectrum of Multi-Channel EMG')
    axs[1].legend(loc='upper right', ncol=2)
    axs[1].grid(True)
    axs[1].set_xlim(0, fs/2)  # Limit to Nyquist frequency

    plt.tight_layout()
    plt.show()