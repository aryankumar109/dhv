import numpy as np 
import matplotlib.pyplot as plt
from numpy import random

def gensin(freq,sample_rate,duration,power=1):
    x= np.linspace(0,duration,(int(sample_rate*duration)))
    y= power*np.sin(2*np.pi*freq*x)
    return x,y
#%%

time = np.linspace(0,4*np.pi,1000)
zline = np.zeros((1000))
signal = 4*np.sin(time)
sig2 = np.sin(time/2)
sig3 = np.sin(time+1)
# #random noise
# noise = (random.rand(1000)-0.5)
# #gaussian noise
# noise2 = random.normal(0,1,1000)/2




plt.figure(figsize=(20,10))
plt.plot(time,signal)
# plt.plot(time,(signal+noise2))
# plt.plot(time,(sig3+signal))
plt.plot(time,zline,'k')


#%%
from scipy.fft import rfft, rfftfreq

freq = 1
sample_rate= 100
duration=3
#total number of samples
N = sample_rate * duration
x, sig4= gensin(freq,sample_rate,duration)
_, sig5= gensin(freq+5,sample_rate,duration,0.5)
#random noise
# noise = (random.rand(N)-0.5)
#gaussian noise
noise2 = random.normal(0,1,N)/2
sig4 = sig4 + noise2 + sig5
plt.figure(figsize=(20,10))
plt.plot(x,sig4)
plt.plot(x,np.zeros((len(x))),'k')


# Number of samples in normalized_tone
N = sample_rate * duration

yf = rfft(sig4)
xf = rfftfreq(int(N), 1./sample_rate)

plt.figure(figsize=(20,10))
plt.plot(xf, np.abs(yf))
plt.show()

#%%
from scipy.fft import irfft
#removing noise
passfreq = 6
filterindex = duration*passfreq

yf[filterindex+1:] = 0
plt.figure(figsize=(20,10))
plt.plot(xf, np.abs(yf))
plt.show()

filt_signal = irfft(yf)
plt.figure(figsize=(20,10))
plt.plot(x,filt_signal)
plt.plot(x,np.zeros((len(x))),'k')


















