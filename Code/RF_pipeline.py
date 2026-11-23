# Libraries
import time
import numpy as np

# Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# My files
import globals
import data_collector
import plots
import filters


def extract_time_features(windows: np.ndarray, database, enabled_features=None):
    # Default: enable all features
    all_features = [
        'mav', 'wl', 'zc', 'ssc', 'rms', 'wamp', 'ld', 'mas', 
        'h_activity', 'h_mobility', 'iemg'
    ]
    
    if enabled_features is None:
        enabled_features = all_features
    else:
        enabled_features = [f for f in enabled_features if f in all_features]

    num_electrodes = globals.SENSORS[database - 1]
    n_windows = windows.shape[0]

    # Transpose signals: (n_windows, 40, num_electrodes) → (n_windows, num_electrodes, 40)
    t = time.time()
    signals = np.array(windows, dtype=np.float64).transpose(0, 2, 1)
    print(f"Array conversion: {time.time() - t:.2f}s")

    feature_dict = {}

    if 'mav' in enabled_features:
        feature_dict['mav'] = np.mean(np.abs(signals), axis=2)
    if 'wl' in enabled_features:
        feature_dict['wl'] = np.sum(np.abs(np.diff(signals, axis=2)), axis=2)
    if 'zc' in enabled_features:
        feature_dict['zc'] = np.sum((signals[:, :, :-1] * signals[:, :, 1:] < 0) &
                                    (np.abs(signals[:, :, :-1] - signals[:, :, 1:]) > 0.0), axis=2)
    if 'ssc' in enabled_features:
        diff = np.diff(signals, axis=2)
        feature_dict['ssc'] = np.sum((diff[:, :, :-1] * diff[:, :, 1:] < 0) &
                                     (np.abs(diff[:, :, :-1]) > 0.0) &
                                     (np.abs(diff[:, :, 1:]) > 0.0), axis=2)
    if 'rms' in enabled_features:
        feature_dict['rms'] = np.sqrt(np.mean(signals ** 2, axis=2))
    if 'wamp' in enabled_features:
        feature_dict['wamp'] = np.sum(np.abs(np.diff(signals, axis=2)) > 0.00, axis=2)
    if 'ld' in enabled_features:
        feature_dict['ld'] = np.exp(np.mean(np.log(np.abs(signals) + 1e-10), axis=2))
    if 'mas' in enabled_features:
        feature_dict['mas'] = np.mean(np.abs(np.diff(signals, axis=2)), axis=2)
    if 'h_activity' in enabled_features:
        feature_dict['h_activity'] = np.var(signals, axis=2)
    if 'h_mobility' in enabled_features:
        feature_dict['h_mobility'] = np.sqrt(np.var(np.diff(signals, axis=2)) / (np.var(signals, axis=2) + 1e-10))
    if 'iemg' in enabled_features:
        feature_dict['iemg'] = np.sum(np.abs(signals), axis=2)

    # Combine selected features per electrode
    n_features = len(enabled_features) * num_electrodes
    features = np.zeros((n_windows, n_features))

    for e in range(num_electrodes):
        per_electrode_feats = [feature_dict[feat][:, e] for feat in enabled_features]
        features[:, e * len(enabled_features):(e + 1) * len(enabled_features)] = np.vstack(per_electrode_feats).T

    # # Normalize combined features (z-score)
    # features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)

    return features

def train(model, train_wdw, train_lbl):
    print(f'Training started...')
    train_start = time.time()

    model.fit(train_wdw, train_lbl)

    training_time = time.time()-train_start
    print(f'Model trained. Training time: {training_time:.2f}s')

    train_preds = model.predict(train_wdw)
    train_acc = accuracy_score(train_lbl, train_preds)
    print(f'Training accuracy: {train_acc*100:.2f}')

    return training_time, train_acc

def test(model, database, test_number, test_wdw, test_lbl):
    start_pred = time.time()

    test_preds  = model.predict(test_wdw)
    acc = accuracy_score(test_lbl, test_preds)

    total_pred_time = time.time() - start_pred
    pred_rate = total_pred_time / test_wdw.shape[0]
    
    print(f'Model tested. Testing time: {total_pred_time:.2f}s')
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Predict Time: {total_pred_time:.2f}s, Predict rate: {pred_rate:.4f}s per input")

    plots.plot_confusion_matrix(labels=test_lbl, predictions=test_preds, database=database, method='RF', test_num=test_number, display=False)

    return acc, total_pred_time, pred_rate    

def pre_process(database, data, labels, repetitions, feature_list=None):
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

    # # Balance datasets
    # # Count of each class label in y
    # class_counts_train = np.bincount(labels_train)
    # class_counts_test  = np.bincount(labels_test)
    # class_counts_val   = np.bincount(labels_val)

    # # Get the index with the smallest count
    # min_class_train = np.argmin(class_counts_train)
    # min_class_test  = np.argmin(class_counts_test)
    # min_class_val   = np.argmin(class_counts_val)

    # # Get the smallest count
    # min_count_train = class_counts_train[min_class_train]
    # min_count_test  = class_counts_test[min_class_test]
    # min_count_val   = class_counts_val[min_class_val]

    # print(f"Smallest train class data: {min_class_train}, Count: {min_count_train}")
    # print(f"Smallest test class data:  {min_class_test}, Count: {min_count_test}")
    # print(f"Smallest val class data:   {min_class_val}, Count: {min_count_val}")

    # # Balance classes
    # target_per_class = min_count_train
    # data_train, labels_train = data_collector.balance_data(x=data_train, y=labels_train, target_per_class=target_per_class, class_counts=class_counts_train)
    # target_per_class = min_count_val
    # data_val, labels_val     = data_collector.balance_data(x=data_val,  y=labels_val,    target_per_class=target_per_class, class_counts=class_counts_val)
    # target_per_class = min_count_test
    # data_test, labels_test   = data_collector.balance_data(x=data_test, y=labels_test,   target_per_class=target_per_class, class_counts=class_counts_test)
    
    # print(f"Balanced sets: Tr{data_train.shape}, Va{data_val.shape}, Te{data_test.shape}")
    plots.plot_class_distribution(train_labels=labels_train, val_labels=labels_val, test_labels=labels_test, database=database, display=False)


    # Extract features
    print(f'Feature extraction started')
    feat_start = time.time()

    data_train = extract_time_features(windows=data_train, database=database, enabled_features=feature_list)
    data_val   = extract_time_features(windows=data_val  , database=database, enabled_features=feature_list)
    data_test  = extract_time_features(windows=data_test , database=database, enabled_features=feature_list)

    feature_time = time.time()-feat_start
    feature_time_rate = feature_time / (data_train.shape[0]+data_test.shape[0]+data_val.shape[0])
    print(f"Feature extraction Time: {feature_time:.2f}s, Feature rate: {feature_time_rate:.2f}s per input")
    print(f"Data shapes. Train: {data_train.shape}. Val: {data_val.shape}. Test: {data_test.shape}")

    # Scale all datasets per channel (fit on training only)
    scale_start = time.time()
    data_train, data_val, data_test, scalars = filters.scale_per_feature(train=data_train, val=data_val, test=data_test)

    scale_time = time.time() - scale_start
    scale_rate = scale_time / (data_train.shape[0]+data_test.shape[0]+data_val.shape[0])
    print(f"Scale time: {scale_time:.2f}s, Scale rate: {scale_rate:.2f}s per input")
    print(f"Scaled sets: Tr{data_train.shape}, Va{data_val.shape}, Te{data_test.shape}")

    return data_train, labels_train, data_val, labels_val, data_test, labels_test, gesture_set_list, feature_time, feature_time_rate

def train_test_rf(database, test_number, data_train, labels_train, data_val, labels_val, data_test, labels_test, type='valid', feature_list=None):
    
    # Model hyperparameters
    n_estimators      = 350
    max_depth         = 60
    min_samples_split = 2
    min_samples_leaf  = 1
    max_features      = 'sqrt'

    # Create model
    # model = RandomForestClassifier(n_estimators=500, max_depth=50, min_samples_split=2, min_samples_leaf=1, n_jobs=-1, bootstrap=True, oob_score=False)
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split, 
        min_samples_leaf=min_samples_leaf, 
        max_features=max_features,
        class_weight='balanced',
        n_jobs=-1, verbose=False, bootstrap=False)

    if type == 'valid':
        # Validation testing
        pass
    elif type == 'test':
        # Final testing
        # Combine train and validation sets
        data_train   = np.concatenate((data_train, data_val))
        labels_train = np.concatenate((labels_train, labels_val))
    else:
        raise Exception

    # Train 
    training_time, train_acc = train(
        model=model, 
        train_wdw=data_train, 
        train_lbl=labels_train)
    
    # Plot feature importances
    importances = model.feature_importances_
    plots.plot_rf_feature_importances(importances=importances, database=database, feature_list=feature_list, test_num=test_number)

    if type == 'valid':
        # Validation test
        test_acc, predict_time, predict_times_rate = test(
            model=model, 
            database=database, 
            test_number=test_number, 
            test_wdw=data_val, 
            test_lbl=labels_val)
    else:
        # Final test
        test_acc, predict_time, predict_times_rate = test(
            model=model, 
            database=database, 
            test_number=test_number, 
            test_wdw=data_test, 
            test_lbl=labels_test)
    
    val_acc = float('nan')

    return train_acc, val_acc, test_acc, training_time, predict_time, predict_times_rate








