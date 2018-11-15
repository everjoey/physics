#!/usr/bin/env python3
import sys
import os
#from scipy import optimize
'''
for i in sys.path:
	print(i)
print(os.system('echo $PATH'))
'''

#p = scipy.poly1d([2,3,4])
#print(p)
#x = optimize.newton(f, 1)
#print(round(x,3))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import xlabel
import scipy.special as sp
import scipy

#signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
#print (signal)
t = np.arange(-100,100,0.1)
#signal = np.array(np.exp(-np.pi*np.square(t)))
#signal = np.exp(-np.pi*(t*t))#np.sinc(t)
signal = 1/np.cosh(t)
signal = np.exp(-np.abs(t))
signal = np.sinc(t)
#signal = sp.jv(0,t)

print(t, len(t))
print(signal)
'''
fourier = np.zeros(len(signal), dtype=complex)
for i in range(len(signal)):
    signal2 = np.zeros(len(signal), dtype=complex)
    for j in range(len(signal)):
        signal2[j] = signal[j]*np.exp(-1j*2*np.pi*(j-1000)*1*(float(i)/len(signal)))
    fourier[i] = np.sum(signal2)
print(fourier)
'''
'''
fourier = np.zeros(len(signal), dtype=complex)
for i in range(len(signal)):
    signal2 = np.zeros(len(signal), dtype=complex)
    for j in range(len(signal)):
        signal2[j] = signal[j]*np.exp(-1j*2*np.pi*(j)*1*(float(i)/len(signal)))
    fourier[i] = np.sum(signal2)
print(fourier)
'''

fourier2 = np.fft.fft(signal)*np.array([np.exp(1j*2*np.pi/len(signal)*len(signal)/2*i) for i in range(len(signal))])
print(fourier2)

ifourier2 = np.fft.ifft(fourier2*np.array([np.exp(-1j*np.pi*i) for i in range(len(signal))]))
print(ifourier2)

freq = np.fft.fftfreq(len(signal))
print(freq)
fig1 = plt.figure()
ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
l1 = ax.plot(t, signal)
#l2 = ax.plot(freq, [np.pi*10*1/np.cosh(np.pi*np.pi*f*10) for f in freq])
#l2 = ax.plot(freq, [2*0.1/(0.01+4*np.pi*np.pi*f*f) for f in freq])
ax.set_xlim(-10, 10)
ax.set_ylim(-1, 1)
#np.cosh(np.pi**2/0.1*2*np.pi*f)

fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
l2 = ax2.plot(freq, fourier2)
ax2.set_xlim(-0.5, 0.5)
ax2.set_ylim(-10, 20)

fig3 = plt.figure()
ax3 = fig3.add_axes([0.1, 0.1, 0.8, 0.8])
l3 = ax3.plot(t, ifourier2)
ax3.set_xlim(-10, 10)
ax3.set_ylim(-1, 1)

plt.show()
