import numpy as np
import os
import time
import concurrent.futures
from scipy.signal import resample 
import cupy as cp
from cupyx.scipy.signal import resample as cp_resample, butter, sosfiltfilt
from numpy.lib.stride_tricks import sliding_window_view
import random
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from collections import defaultdict

import globals
import filters
import mat_file

def find_average_big_window_width(data, labels, repetitions):
    # Initialize dictionaries to store lengths per class and repetition
    class_lengths = defaultdict(list)
    repetition_lengths = defaultdict(list)

    # Iterate through all windows
    for window, class_label, repetition_label in zip(data, labels, repetitions):
        window_length = len(window)  # assuming window is shape (time, features)
        class_lengths[class_label].append(window_length)
        repetition_lengths[repetition_label].append(window_length)

    # Compute averages
    avg_class_lengths = {label: np.mean(lengths) for label, lengths in class_lengths.items()}
    avg_repetition_lengths = {label: np.mean(lengths) for label, lengths in repetition_lengths.items()}

    # Display results
    print("Average window width per class label:")
    for label, avg_len in avg_class_lengths.items():
        print(f"Class {label}: {avg_len:.2f} time steps")

    print("\nAverage window width per repetition label:")
    for label, avg_len in avg_repetition_lengths.items():
        print(f"Repetition {label}: {avg_len:.2f} time steps")

def collect_global_scalars_from_file(database):
    # Set your data directory
    # Different naming convention for DB1 and others
    data_dir = globals.PROJECT_DIR / f'Ninapro_data/DB{database}'
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mat')]

    # Gather all EMG data
    all_emg = []

    for file in file_list:
        _, _, emg, *_ = mat_file.extract_mat_file(file)
        all_emg.append(emg)  # shape: (samples, channels)

    # Combine into one big array: shape (total_samples, channels)
    combined_emg = np.vstack(all_emg)

    # Compute global stats
    global_mean = np.mean(combined_emg, axis=0).tolist()
    global_std  = np.std(combined_emg, axis=0).tolist()
    global_min  = np.min(combined_emg, axis=0).tolist()
    global_max  = np.max(combined_emg, axis=0).tolist()

    # Print in copy-pasteable format
    print(f"\n# 📊 Stats for Dataset {database}")
    print("MEAN =", global_mean)
    print("STD  =", global_std)
    print("MAX  =", global_max)
    print("MIN  =", global_min)

def collect_global_scalars_from_windows(windows):
    # Combine all windows
    all_windows = []
    for wdw in windows:
        all_windows.append(wdw)
    
    # Convert to np for stats
    all_windows = np.vstack(all_windows)

    # Compute global stats
    global_mean = np.mean(all_windows, axis=0)
    global_std  = np.std(all_windows, axis=0)
    global_min  = np.min(all_windows, axis=0)
    global_max  = np.max(all_windows, axis=0)
    
    return global_mean, global_std, global_min, global_max

def modify_label(label, database, exercise):
    # Label mapping
    if database in [2, 3, 7]:
        label += 11
    elif database == 1:
        if exercise == 1:
            label -= 1
        elif exercise == 2:
            label += 11
        elif exercise == 3:
            label += 28
    elif database == 4:
        if exercise == 2:
            label -= 1
        elif exercise == 1:
            label += 11
        elif exercise == 3:
            label += 28
    elif database == 5:
        if exercise == 3:
            label -= 1
        elif exercise == 1:
            label += 11
        elif exercise == 2:
            label += 28
    else:
        raise ValueError(f"Unsupported database: {database}")

    if label < 0 or label > 60:
        raise ValueError(f"Invalid label: {label}")
    
    return label

def balance_data(x, y, target_per_class, class_counts):
    
    # Initialise new lists
    x_balanced, y_balanced = [], []

    # Loop through all classes with a non-zero count
    for cls in np.nonzero(class_counts)[0]:
        cls_indices = np.where(y == cls)[0]
        cls_count = len(cls_indices)
        if cls_count == 0:
            continue
        if cls_count >= target_per_class:
            # Randomly sample target number without duplicates
            sampled_indices = np.random.choice(cls_indices, target_per_class, replace=False)
        else:
            # Randomly sample with duplicates (oversample)
            sampled_indices = np.random.choice(cls_indices, target_per_class, replace=True)

        x_balanced.append(x[sampled_indices])
        y_balanced.append(np.full(target_per_class, cls))

    x = np.vstack(x_balanced)
    y = np.hstack(y_balanced) 

    return x, y

def separate_repetitions_train_valid_test(database, windows, labels, rep_nums, random_sets=False, proportions = [0.7, 0.1, 0.2]):
    train_windows, train_labels = [], [] 
    val_windows, val_labels     = [], []
    test_windows, test_labels   = [], []

    if random_sets:
        # Global variable for number of gestures
        num_gestures = globals.REPS[database-1]

        # Ensure they sum to 1.0
        assert sum(proportions) == 1.0, "Proportions must sum to 1.0"

        # Create a list of gesture indices
        gesture_indices = list(range(1, num_gestures + 1))
        # Shuffle in place
        random.shuffle(gesture_indices)  

        # Calculate split points
        split1 = int(proportions[0] * num_gestures)
        split2 = split1 + int(proportions[1] * num_gestures)
        # split2 = split1 + max(int(proportions[1] * num_gestures), 1)

        # Split the shuffled list into 3 subsets
        train_reps  = gesture_indices[:split1]
        val_reps    = gesture_indices[split1:split2]
        test_reps   = gesture_indices[split2:]

        # Sort the lists into ascending order (for readability)
        train_reps.sort()
        val_reps.sort()
        test_reps.sort()

        print(f'Random gesture sets. Tr={train_reps}, Vd={val_reps}, Te={test_reps}')
    else:
        train_reps, val_reps, test_reps = globals.ML_REP_SETS_3[database-1]
        print(f'Preset gesture sets. Tr={train_reps}, Vd={val_reps}, Te={test_reps}')

    for win, label, rep in zip(windows, labels, rep_nums):
        if rep in train_reps:
            train_windows.append(win)
            train_labels.append(label)
        elif rep in val_reps:
            val_windows.append(win)
            val_labels.append(label)
        elif rep in test_reps:
            test_windows.append(win)
            test_labels.append(label)
        else:
            raise Exception

    return train_windows, train_labels, train_reps, val_windows, val_labels, val_reps, test_windows, test_labels, test_reps

def resample_and_subdivide_np(database, windows, labels, orig_fs, target_fs=200.0, window_width_ms=200.0, overlap_percent=50):
    """Resample windows to target_fs and split into smaller overlapping windows of shape (40, 10)."""
    window_samples = int(window_width_ms * target_fs / 1000)    # e.g., 200ms = 40 samples
    stride = int(window_samples * (1 - overlap_percent / 100))  # e.g., 50% overlap = stride of 20

    small_windows = []
    small_labels = []

    for win, lbl in zip(windows, labels):
        # Shape: (original_samples, num_electrodes)
        num_resampled = int(round(win.shape[0] * target_fs / orig_fs))
        # Shape: (resampled_samples, num_electrodes)
        win_resampled = resample(win, num_resampled, axis=0)  

        # Skip if too short
        if win_resampled.shape[0] < window_samples:
            continue

        # Create sliding windows: shape -> (num_segments, num_electrodes, window_samples)
        segments = sliding_window_view(win_resampled, window_shape=(window_samples), axis=0)
        segments = segments[::stride]
        # shape -> (num_segments, window_samples, num_electrodes)
        segments = segments.transpose(0, -1, -2)

        small_windows.append(segments)
        small_labels.extend([lbl] * segments.shape[0])

    if small_windows:
        small_windows = np.concatenate(small_windows, axis=0)
    else:
        num_electrodes = globals.SENSORS[database-1]
        small_windows = np.empty((0, window_samples, num_electrodes))

    small_labels = np.array(small_labels)
    return small_windows, small_labels

def extract_big_windows(emg_data: np.ndarray, stimulus: np.ndarray, repetition: np.ndarray, database: int, exercise: int):
    """Extracts full contiguous positive segments from trigger."""
    # Use stimulus as the mask to identify emg activity
    stimulus = stimulus.flatten()
    indices = np.where(stimulus > 0)[0]
    if len(indices) == 0:
        return [], []
    # Start of window: 0 to class int
    starts  = indices[np.insert(np.diff(indices) > 1, 0, True)]
    # End of window: class int to 0
    ends    = indices[np.append(np.diff(indices) > 1, True)]

    windows, labels, repetitions = [], [], []
    for start, end in zip(starts, ends):
        # Stimulus contains gesture # = label
        label    = stimulus[start]
        # Record repetition to separate later
        rep_num  = repetition[start]
        # Modify label based of database and exercise naming conventions
        label = modify_label(label=label, database=database, exercise=exercise)

        windows.append(emg_data[start:end])
        labels.append(label)
        repetitions.append(rep_num)

    return windows, labels, repetitions

def file_extractor(args):
    database, i, j, pre_standardise = args
    # Different naming convention for DB1 and others
    if database == 1:
        file_str = f'Ninapro_data/DB{database}/S{i+1}_A1_E{j+1}.mat'
    else:
        file_str = f'Ninapro_data/DB{database}/S{i+1}_E{j+1}_A1.mat'

    # File directory
    data_path = globals.PROJECT_DIR / file_str
    subject, exercise, emg, stimulus, restimulus, repetition, rerepetition = mat_file.extract_mat_file(data_path)
    
    
    # Convert to NumPy for filtering
    emg = np.array(emg)
    sampling_f = globals.FS[database-1]

    emg = filters.apply_lowpass_filter(data=emg, fs=sampling_f, cutoff=1.0)

    # # If 500Hz is greater than Nyquist (cant apply lowpass filter)
    # if 500 > sampling_f/2:
    #     # Can only apply highpass - remove motion artefacts
    #     # Highpass above 20 Hz 
    #     cutoff = 20.0
    #     sos_filter = butter(4, cutoff, btype='high', fs=sampling_f, output='sos')

    #     for ch in range(emg.shape[1]):
    #         emg[:, ch] = sosfiltfilt(sos_filter, emg[:, ch])  # Apply lowpass first
    # else:
    #     # Bandpass 20Hz to 500Hz (EMG Range)
    #     lowcut  = 20.0
    #     highcut = 500.0
    #     # Design filters
    #     sos_bandpass = butter(4, [lowcut, highcut], btype='band', fs=sampling_f, output='sos')

    #     for ch in range(emg.shape[1]):
    #         # Bandpass filter (using SOS)
    #         emg[:, ch] = sosfiltfilt(sos_bandpass, emg[:, ch])


    if pre_standardise:
        # Local standardise
        # emg = (emg - cp.mean(emg, axis=0)) / (cp.std(emg, axis=0) + 1e-8)

        # Global standardise
        index = database - 1
        global_mean = np.array(globals.MEANS[index])
        global_std  = np.array(globals.STDS[index])
        
        emg = (emg - global_mean) / (global_std + 1e-8)

        # # Global normalization
        # index = database - 1
        # global_min = np.array(globals.MINS[index])
        # global_max = np.array(globals.MAXS[index])
        
        # emg = (emg - global_min) / (global_max - global_min + 1e-8)  # Normalization to [0, 1]

    # Move to GPU for extraction
    emg = cp.asnumpy(emg)
    # Extract large windows
    large_windows, labels, repetitions = extract_big_windows(emg_data=emg, stimulus=restimulus, repetition=rerepetition, database=database, exercise=exercise)

    return large_windows, labels, repetitions

def collect_all_big_windows(database, pre_standardise=False):
    """
    Collect all windows across subjects and exercises for a given database using restimulus mask.
    
    Args:
        database (int): NinaPro database number (1-5, 7)
    
    Returns:
        all_windows (list): List of windowed EMG data
        all_labels (list): Corresponding labels
        all_repetitions (list): Corresponding repetition (for train/ val/ test split)
    """
    all_windows, all_labels, all_repetitions = [], [], []
    
    # Prepare arguments for multiprocessing
    args_list = [
        (database, i, j, pre_standardise)
        for i in range(globals.SUBJECTS[database-1])
        for j in range(globals.EXERCISES[database-1])
    ]

    # Debug
    # result = file_extractor(args=[database, 0, 0, pre_standardise])

    max_workers = 6
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(file_extractor, args_list))

    # Aggregate results
    for windows, labels, repetition in results:
        all_windows.extend(windows)
        all_labels.extend(labels)
        all_repetitions.extend(repetition)

    return all_windows, all_labels, all_repetitions

def process_file(args):
    # Extract variables from argument list
    database, file = args

    # Extract elements from file
    subject, exercise, emg, stimulus, restimulus, repetition, rerepetition = mat_file.extract_mat_file(file)
    
    # Convert to NumPy for speed
    emg = np.array(emg)
    fs = globals.FS[database-1]

    # Filter emg
    # emg = filters.apply_lowpass_filter(data=emg, fs=fs, cutoff=1.0)

    # if database in [1,5]:
    #     emg = filters.apply_highpass_filter(data=emg, fs=fs, cutoff=20.0)
    # else:
    #     emg = filters.apply_notch_filter(data=emg, fs=fs, freq=50.0)
    #     # emg = filters.apply_bandpass_filter(data=emg, fs=fs, lowcut=20.0, highcut=500.0)

    # emg = filters.full_wave_rectify(data=emg)


    if database in [1, 5]:
        # Already filtered for powerline
        # Already filtered for EMG range
        # emg = filters.apply_lowpass_filter(data=emg, fs=fs, cutoff=1.0)
        emg = filters.rms_rectify(data=emg, window_size=int(100e-03*fs))
    else:
        # Filter powerline
        emg = filters.apply_notch_filter(data=emg, fs=fs, freq=50.0)
        # Filter for EMG range (<500Hz)
        # emg = filters.apply_lowpass_filter(data=emg, fs=fs, cutoff=500.0)
        # Rectify emg
        emg = filters.rms_rectify(data=emg, window_size=int(100e-03*fs))
    
    # Extract large windows using restimulus as the mask
    large_windows, labels, repetitions = extract_big_windows(emg_data=emg, stimulus=restimulus, repetition=rerepetition, database=database, exercise=exercise)

    return large_windows, labels, repetitions

def multi_process_extract_big_windows(database):
    # Prepare arguments for multiprocessing
    data_dir = globals.PROJECT_DIR / f'Ninapro_data/DB{database}'
    args_list = [
        (database, os.path.join(data_dir, f)) 
        for f in os.listdir(data_dir) if f.endswith('.mat')
        ]

    print(f'Database: {database}. Files found: {len(args_list)}')

    # Initialise results
    all_windows, all_labels, all_repetitions = [], [], []
    
    # Parallel processing each file - collect all results
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(process_file, args_list))

    # Aggregate results
    for windows, labels, repetition in results:
        all_windows.extend(windows)
        all_labels.extend(labels)
        all_repetitions.extend(repetition)

    return all_windows, all_labels, all_repetitions