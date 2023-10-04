# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:03:42 2023

@author: B21121 Ronak,
         B21282 Atharva
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set parameters for the signal
amp = 1       # Amplitude   
freq=int(input("Input Frequency: "))   # Frequency in Hz
phase = np.pi/2  # Phase shift in radians

# Generate the time axis
t = np.linspace(0, 1, 1000)

# Generate the signal
sig = amp * np.sin(2*np.pi*freq*t + phase)

# Add noise to the signal
noise = np.random.normal(0, 0.3, len(sig))
noisy_signal = sig + noise

# Define the filter
fs = 1000 # Sampling frequency
cutoff = 30 # Cutoff frequency of the filter
nyquist = 0.5*fs
cutoff_norm = cutoff/nyquist
b, a = signal.butter(4, cutoff_norm, 'lowpass')

# Apply the filter
filtered_signal = signal.filtfilt(b, a, noisy_signal)

# Plot the signals
plt.plot(t, sig, label='Signal')
plt.plot(t, noisy_signal, label='Noisy signal')
plt.plot(t, filtered_signal, label='Filtered signal')
plt.legend()
plt.show()