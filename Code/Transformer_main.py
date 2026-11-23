# Libraries
import torch
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='cupyx.jit.rawkernel')

# My files
import data_collector
import documentation

import Transformer_pipeline


if __name__ == "__main__":  # Ensure multiprocessing runs safely on Windows
    print(torch.__version__)          
    print(torch.cuda.is_available())  
    print(torch.version.cuda)         
    print(torch.cuda.get_device_name(0))  

    METHOD = 'Transformer'

    # database = [1, 2, 3, 4, 5, 7]
    database = [5]
    num_tests = 1

    # plots.plot_all_raw_data(database=4)

    # Initialise lists to track gesture -> accuracies (quality assessment)
    all_accuracies = []
    all_gest_sets  = []

    for db in database:
        # Record individual test time
        start = time.time()

        # Collect big windows from DB files
        data, labels, repetitions = data_collector.multi_process_extract_big_windows(database=db)
        print(f'Big windows collected')
        print(f"Collection time: {time.time() - start:.2f}s")
        print(f'Windows: {len(data)}, Labels: {len(labels)}, Rep_nums: {len(repetitions)}')
        
        # Normalize labels to start from 0 
        print(f'Original training labels: {min(labels)}-{max(labels)}')
        labels = np.array(labels) - np.min(labels)
        print(f'Adjusted training labels: {min(labels)}-{max(labels)}')

        # plots.plot_all_windows(data=data, labels=labels)

        # Initialise log variables
        train_accuracies = []
        val_accuracies   = []
        test_accuracies  = []
        train_times = []
        predict_times = []
        predict_times_rates = []
        gesture_sets = []

        for i in range(num_tests):

            data_train, labels_train, data_val, labels_val, data_test, labels_test, gesture_set_list = Transformer_pipeline.pre_process(
                database=db, 
                data=data, 
                labels=labels, 
                repetitions=repetitions)
            
            train_acc, val_acc, test_acc, training_time, predict_time, predict_times_rate = Transformer_pipeline.train_test_transformer(
                database=db, 
                test_number=i, 
                data_train=data_train, 
                labels_train=labels_train, 
                data_val=data_val, 
                labels_val=labels_val, 
                data_test=data_test, 
                labels_test=labels_test)

            # Store all results per test
            gesture_sets.append(gesture_set_list)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            test_accuracies.append(test_acc)
            train_times.append(training_time)
            predict_times.append(predict_time)
            predict_times_rates.append(predict_times_rate)
        
        total_time = time.time()-start
        print(f'Total time: {total_time:.2f}s')

        # Aggreagate all 3 accuuracy types [tr, val, te]
        combined_acc = [train_accuracies, val_accuracies, test_accuracies]

        # Log results
        log_file = f'Code/Test_results/{METHOD}/test_results_DB{db}_{METHOD}.txt'
        documentation.log(database=db, file=log_file, gesture_sets=gesture_sets, accuracies=combined_acc, train_times=train_times, predict_times=predict_times, prates=predict_times_rates, total_time=total_time)

        # Log accuracies and rep sets for across all db
        all_accuracies.append(combined_acc)
        all_gest_sets.append(gesture_sets)

    log_file = f'Code/Test_results/{METHOD}/test_results_gesture_vs_acc_{METHOD}.txt'
    documentation.log_gestures_accuracy(file=log_file, all_acc=all_accuracies, all_sets=all_gest_sets)
    log_file = f'Code/Test_results/{METHOD}/test_results_acc_summary_{METHOD}.txt'
    documentation.log_test_accuracies_per_db(file=log_file, all_acc=all_accuracies, database=database)

