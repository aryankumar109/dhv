# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:13:27 2023

@author: B21121 Ronak,
         B21282 Atharva
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the time range for the signal
t = np.linspace(0, 1, 1000)

l=int(input("Low Freq = "))
h=int(input("High Freq = "))
# Generate a mix of low and high frequencies
low_freq = 5 * np.sin(2 * np.pi * l * t)
high_freq = 2 * np.sin(2 * np.pi * h * t)
signal = low_freq + high_freq

# Add some noise to the signal
noise = 0.2 * np.random.normal(size=len(t))
signal = signal + noise

# Perform Fourier transform on the signal
freq = np.fft.fftfreq(len(t), t[1]-t[0])
fft_signal = np.fft.fft(signal)

# Define a band-pass filter to isolate the low-frequency range
low_pass_filter = np.zeros_like(fft_signal)
low_pass_filter[np.abs(freq) < 20] = 1

# Apply the filter to the Fourier transform of the signal
filtered_fft_signal = fft_signal * low_pass_filter

# Perform inverse Fourier transform to get back the filtered signal
filtered_signal = np.fft.ifft(filtered_fft_signal)

# Plot the original signal and the filtered signal
plt.plot(t, signal, label='Original Signal')
plt.plot(t, filtered_signal, label='Filtered Signal')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()