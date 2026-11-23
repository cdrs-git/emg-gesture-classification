import torch
import torch.nn as nn
import random
import numpy as np

# Augmentation Functions
def time_warp(x):
    """Randomly stretch or compress the time axis"""
    original_len = x.shape[0]
    warp_factor = random.uniform(0.9, 1.1)  # Randomly stretch/compress the time series
    warped_len  = int(original_len * warp_factor)
    warped_x = np.interp(np.linspace(0, original_len - 1, warped_len ), np.arange(original_len), x) 
    # Resample back to original length
    resampled_x = np.interp(np.linspace(0, warped_len - 1, original_len), np.arange(warped_len), warped_x)

    return torch.tensor(resampled_x, dtype=torch.float32)

def add_noise(x):
    """Add random noise to the signal"""
    noise_factor = random.uniform(0.05, 0.10)
    noise = np.random.randn(*x.shape) * noise_factor
    return (x + noise).clone().detach().float()

def random_scale(x):
    """Randomly scale the signal by a factor"""
    scale_factor = random.uniform(0.8, 1.2)
    return (x * scale_factor).clone().detach().float()

def time_shift(x):
    """Randomly shift the time series"""
    time_steps = x.shape[0]
    shift_size = int(time_steps * 0.2)
    shift = random.randint(-shift_size, shift_size)
    return torch.tensor(np.roll(x, shift, axis=0)).float()

def random_erasing(x):
    """Randomly erase (zero out) part of the signal"""
    time_steps = x.shape[0]
    erase_size = int(time_steps * 0.1)
    start_idx = random.randint(0, x.shape[0] - erase_size)
    x[start_idx:start_idx + erase_size] = 0  # Erase erase_size consecutive time steps
    return x

# Combine augmentations into a single transform
def time_series_augmentation(x):
    # if random.random() > 0.5:
    #     x = time_warp(x)
    if random.random() > 0.5:
        x = add_noise(x)
    # if random.random() > 0.5:
    #     x = random_scale(x)
    # if random.random() > 0.5:
    #     x = time_shift(x)
    # if random.random() > 0.5:
    #     x = random_erasing(x)
    return x
