import os
import sys
import torch
import random
import time
import cupy
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='cupyx.jit.rawkernel')
import numpy as np
import matplotlib.pyplot as plt


import globals
import mat_file
import data_collector
import Code.Old.sampling as sampling
import plots
import Code.Old.windows as windows
import augmentations
import filters
import documentation

if __name__ == "__main__":  # Ensure multiprocessing runs safely on Windows

    print(torch.__version__)          
    print(torch.cuda.is_available())  
    print(torch.version.cuda)         
    print(torch.cuda.get_device_name(0))  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["OMP_NUM_THREADS"] = "4" # Set numpy threads

    # database = [1, 2, 3, 4, 5, 7]
    database = [1]
    num_tests = 2

    

    plots.plot_all_raw_data(database=1)

    
    # for db in database:



    #     # Collect big windows from DB files
    #     data, labels, repetitions = data_collector.multi_process_extract_big_windows(database=db)
    #     print(f'Big windows collected')
    #     print(f'Windows: {len(data)}, Labels: {len(labels)}, Rep_nums: {len(repetitions)}')
        
    #     data_collector.find_average_big_window_width(data=data, labels=labels, repetitions=repetitions)


        # subject, exercise = 2, 2

        # if db == 1:
        #     file_str = f'Ninapro_data/DB{db}/S{subject}_A1_E{exercise}.mat'
        # else:
        #     file_str = f'Ninapro_data/DB{db}/S{subject}_E{exercise}_A1.mat'

        # data_path = globals.PROJECT_DIR / file_str
        # _, _, emg, *_ = mat_file.extract_mat_file(data_path)

        # emg = np.array(emg)
        # fs=globals.FS[db-1]

        # # if db in [1,5]:
        # #     modified_emg = filters.apply_highpass_filter(data=emg, fs=fs, cutoff=20.0)
        # # else:
        # #     modified_emg = filters.apply_notch_filter(data=emg, fs=fs, freq=50.0)
        # #     modified_emg = filters.apply_bandpass_filter(data=modified_emg, fs=fs, lowcut=20.0, highcut=500.0)
        
        


        # emg = filters.rms_rectify(data=emg, window_size=int(100e-03*fs))

        # modified_emg = filters.full_wave_rectify(data=emg)

        # plots.plot_emg_comparison(emg1=emg, emg2=modified_emg)

        # freqs, fft_magnitudes = filters.compute_multichannel_fft(data=emg, sampling_rate=globals.FS[db-1], apply_window=True)
        # freqs2, fft_magnitudes2 = filters.compute_multichannel_fft(data=modified_emg, sampling_rate=globals.FS[db-1], apply_window=True)

        # plots.plot_fft_comparison(channels=emg.shape[1], fs=globals.FS[db-1], freqs=freqs, freqs2=freqs2, fft_magnitudes=fft_magnitudes, fft_magnitudes2=fft_magnitudes2)