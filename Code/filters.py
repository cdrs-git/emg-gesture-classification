# Numbers and math
import numpy as np
import scipy.signal as signal
from scipy.signal import butter, sosfilt
from scipy.signal import iirnotch, filtfilt

from sklearn.preprocessing import StandardScaler

def lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter to multi-sensor time series data.
    
    Parameters:
        data (ndarray): Input data of shape (1000, 10)
        cutoff (float): Cutoff frequency as a fraction of the Nyquist frequency (0 < cutoff < 0.5*fs)
        fs (float): Sampling frequency
        order (int): Order of the filter
    
    Returns:
        ndarray: Filtered data of shape (1000, 10)
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Define a function to apply the filter to a single sensor
    def apply_lowpass_filter(sensor_data):
        return signal.filtfilt(b, a, sensor_data)
    
    # Apply the filter to each sensor separately
    filtered_data = np.apply_along_axis(apply_lowpass_filter, axis=0, arr=data)
    
    return filtered_data

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to multi-channel time series data.
    
    Parameters:
    -----------
    data : ndarray
        Input data array of shape (n_samples, n_channels)
    lowcut : float
        Lower cutoff frequency in Hz
    highcut : float
        Upper cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int, optional
        Order of the Butterworth filter (default=4)
    
    Returns:
    --------
    filtered_data : ndarray
        Filtered data with same shape as input (n_samples, n_channels)
    """
    
    # Ensure data is 2D and float type
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("Input data must be 2D (n_samples, n_channels)")
    
    n_samples, n_channels = data.shape
    
    # Normalize cutoff frequencies
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Initialize output array
    filtered_data = np.zeros_like(data)
    
    # Apply filter to each channel
    for ch in range(n_channels):
        filtered_data[:, ch] = signal.filtfilt(b, a, data[:, ch])
    
    return filtered_data

def independant_normalise(data):
    eps = 1e-8  # Small value to avoid division by zero
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + eps)
    return data

def standardize(data):
    eps = 1e-8
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + eps
    return (data - mean) / std

def standardize_per_sensor(data, type='indiv'):

    if type == 'indiv':
        # Compute the mean and std for each window independently, across the 40 timesteps (axis 1)
        mean = np.mean(data, axis=1, keepdims=True)  # Shape: (samples, 1, sensors)
        std = np.std(data, axis=1, keepdims=True)    # Shape: (samples, 1, sensors)

        # Standardize the data using the mean and std for each window independently
        standardized_data = (data - mean) / std

        return standardized_data
    elif type == 'global':
        # Data is your 3D numpy array of shape (samples, sequence_length, sensors)
        samples = data.shape[0]
        sequence_length = data.shape[1]
        sensors = data.shape[2]
        # Flatten the data to combine all windows and timesteps (shape: (samples * sequence_length, sensors))
        flattened_data = data.reshape(-1, sensors)  

        # Compute the mean and std for each sensor (across the entire flattened data)
        mean = np.mean(flattened_data, axis=0)  # Shape: (sensors,)
        std  = np.std(flattened_data, axis=0)   # Shape: (sensors,)

        # Standardize the data using the global mean and std for each sensor
        standardized_data = (flattened_data - mean) / std

        # Reshape the data back to the original shape (samples, sequence_length, sensors)
        standardized_data = standardized_data.reshape(samples, sequence_length, sensors)

        return standardized_data
    else:
        return None
    
def normalize_per_sensor(data, type='indiv'):
    if type == 'indiv':
        # Compute the min and max for each window independently, across the 40 timesteps (axis 1)
        min_vals = np.min(data, axis=1, keepdims=True)      # Shape: (samples, 1, sensors)
        max_vals = np.max(data, axis=1, keepdims=True)      # Shape: (samples, 1, sensors)

        normalised_data = (data - min_vals) / (max_vals - min_vals + 1e-8)  # Avoid division by zero

        return normalised_data
    elif type == 'global':
        # Data is your 3D numpy array of shape (samples, sequence_length, sensors)
        samples = data.shape[0]
        sequence_length = data.shape[1]
        sensors = data.shape[2]
        # Flatten the data to combine all windows and timesteps (shape: (samples * sequence_length, sensors))
        flattened_data = data.reshape(-1, sensors)  

        # Compute the min and max for each sensor (across the entire flattened data)
        min_vals = np.min(flattened_data, axis=0)    # Shape: (sensors,)
        max_vals = np.max(flattened_data, axis=0)    # Shape: (sensors,)

        # Standardize the data using the global mean and std for each sensor
        normalised_data = (data - min_vals) / (max_vals - min_vals + 1e-8)

        # Reshape the data back to the original shape (samples, sequence_length, sensors)
        normalised_data = normalised_data.reshape(samples, sequence_length, sensors)

        return normalised_data
    else:
        return None
    
def scale_per_feature(train, val, test):
    num_features = train.shape[1]  # Assuming shape = [samples, features]
    scalers = []
    scaled_train = np.empty_like(train)
    scaled_val = np.empty_like(val)
    scaled_test = np.empty_like(test)

    for f in range(num_features):
        scaler = StandardScaler()
        # Fit on training data only
        scaled_train[:, f] = scaler.fit_transform(train[:, f].reshape(-1, 1)).flatten()
        scaled_val[:, f] = scaler.transform(val[:, f].reshape(-1, 1)).flatten()
        scaled_test[:, f] = scaler.transform(test[:, f].reshape(-1, 1)).flatten()
        scalers.append(scaler)

    return scaled_train, scaled_val, scaled_test, scalers

def scale_per_channel(train, val, test):
    num_channels = train.shape[2]
    scalers = []
    scaled_train = np.empty_like(train)
    scaled_val   = np.empty_like(val)
    scaled_test  = np.empty_like(test)

    for c in range(num_channels):
        scaler = StandardScaler()
        # Fit on training data only
        scaled_train[:, :, c] = scaler.fit_transform(train[:, :, c])
        scaled_val [:, :, c]  = scaler.transform(val[:, :, c])
        scaled_test[:, :, c]  = scaler.transform(test[:, :, c])
        scalers.append(scaler)

    return scaled_train, scaled_val, scaled_test, scalers

def global_scale_all_windows(windows, mean, std, max, min, type='Standardise'):
    pass

def compute_multichannel_fft(data, sampling_rate, apply_window=True):
    """
    Compute the FFT for a multi-channel signal and optionally plot the frequency spectrum.
    
    Parameters:
    - data: 2D NumPy array (n_samples, n_channels) where each column is a channel
    - sampling_rate: Sampling frequency in Hz (e.g., 2000 Hz for Ninapro DB1)
    - apply_window: Boolean to apply a Hann window before FFT (default: True)
    - plot: Boolean to plot the frequency spectrum (default: True)
    
    Returns:
    - freqs: Array of frequency bins (Hz)
    - fft_magnitudes: 2D array of FFT magnitudes (n_freqs, n_channels)
    """
    # Get dimensions
    n_samples, n_channels = data.shape
    
    # Apply windowing to reduce spectral leakage (optional)
    if apply_window:
        window = np.hanning(n_samples)
        data = data * window[:, np.newaxis]  # Apply window to each channel
    
    # Compute FFT for each channel
    fft_data = np.fft.fft(data, axis=0)
    
    # Calculate single-sided amplitude spectrum (positive frequencies only)
    n_freqs = n_samples // 2 + 1
    fft_magnitudes = np.abs(fft_data[:n_freqs, :]) / n_samples  # Normalize by n_samples
    fft_magnitudes[1:-1, :] *= 2  # Double non-DC/nyquist components for single-sided spectrum
    
    # Compute frequency bins
    freqs = np.fft.fftfreq(n_samples, d=1/sampling_rate)[:n_freqs]
    
    return freqs, fft_magnitudes

def apply_notch_filter(data, fs, freq=50.0, Q=30.0):
    """
    Apply a notch filter to multi-channel data (samples x channels).

    Parameters:
        data (np.ndarray): 2D array of shape (n_samples, n_channels)
        fs (float): Sampling frequency in Hz
        freq (float): Frequency to notch (e.g., 50 or 60 Hz)
        Q (float): Quality factor

    Returns:
        np.ndarray: Filtered data with the same shape
    """
    if data.ndim != 2:
        raise ValueError("Input data must be 2D (samples x channels)")

    b, a = iirnotch(freq, Q, fs)
    filtered_data = np.zeros_like(data)

    for ch in range(data.shape[1]):
        filtered_data[:, ch] = filtfilt(b, a, data[:, ch])

    return filtered_data

def apply_lowpass_filter(data, fs, cutoff=1.0, order=4):
    """
    Apply a low-pass Butterworth filter to multi-channel data (samples x channels).

    Parameters:
        data (np.ndarray): 2D array of shape (n_samples, n_channels)
        fs (float): Sampling frequency in Hz
        cutoff (float): Cutoff frequency in Hz
        order (int): Order of the filter (higher = steeper roll-off)

    Returns:
        np.ndarray: Filtered data with the same shape
    """
    if data.ndim != 2:
        raise ValueError("Input data must be 2D (samples x channels)")

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = np.zeros_like(data)

    for ch in range(data.shape[1]):
        filtered_data[:, ch] = filtfilt(b, a, data[:, ch])

    return filtered_data

def apply_highpass_filter(data, fs, cutoff=20.0, order=4):
    """
    Apply a high-pass Butterworth filter to multi-channel data (samples x channels).

    Parameters:
        data (np.ndarray): 2D array of shape (n_samples, n_channels)
        fs (float): Sampling frequency in Hz
        cutoff (float): Cutoff frequency in Hz
        order (int): Order of the filter (higher = steeper roll-off)

    Returns:
        np.ndarray: Filtered data with the same shape
    """
    if data.ndim != 2:
        raise ValueError("Input data must be 2D (samples x channels)")

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = np.zeros_like(data)

    for ch in range(data.shape[1]):
        filtered_data[:, ch] = filtfilt(b, a, data[:, ch])

    return filtered_data

def apply_bandpass_filter(data, fs, lowcut=20.0, highcut=500.0, order=4):
    """
    Apply a band-pass Butterworth filter to multi-channel data (samples x channels).

    Parameters:
        data (np.ndarray): 2D array of shape (n_samples, n_channels)
        fs (float): Sampling frequency in Hz
        lowcut (float): Low cutoff frequency in Hz
        highcut (float): High cutoff frequency in Hz
        order (int): Order of the filter

    Returns:
        np.ndarray: Filtered data with the same shape
    """
    if data.ndim != 2:
        raise ValueError("Input data must be 2D (samples x channels)")

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = np.zeros_like(data)

    for ch in range(data.shape[1]):
        filtered_data[:, ch] = filtfilt(b, a, data[:, ch])

    return filtered_data

def rms_rectify(data, window_size=100):
    """
    Apply RMS rectification to multi-channel data (samples x channels).

    Parameters:
        data (np.ndarray): 2D array of shape (n_samples, n_channels)
        window_size (int): Size of the moving window for RMS in samples

    Returns:
        np.ndarray: RMS-rectified signal with the same shape
    """
    if data.ndim != 2:
        raise ValueError("Input data must be 2D (samples x channels)")

    n_samples, n_channels = data.shape
    rms_data = np.zeros_like(data)

    for ch in range(n_channels):
        squared = data[:, ch] ** 2
        # Use convolution to compute moving average of squared signal
        kernel = np.ones(window_size) / window_size
        moving_avg = np.convolve(squared, kernel, mode='same')
        rms_data[:, ch] = np.sqrt(moving_avg)

    return rms_data

def full_wave_rectify(data):
    """
    Apply full-wave rectification (absolute value) to multi-channel data.
    Input shape: (samples x channels)
    """
    return np.abs(data)