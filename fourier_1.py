#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def ft(t, f_t):
	F_f = np.fft.fft(f_t)*np.array([np.exp(1j*np.pi*i) for i in range(len(t))])
	f = np.fft.fftfreq(len(t))
	return f, F_f

if __name__ == '__main__':
	t = np.arange(-100, 100, 0.1)
	f_t = np.sinc(t)*np.sin(2*np.pi*1*t)
	f, F_f = ft(t, f_t)

	fig1 = plt.figure()
	ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
	l1 = ax1.plot(t, f_t)
	ax1.set_xlim(-10, 10)
	ax1.set_ylim(-1, 1)

	fig2 = plt.figure()
	ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
	l2 = ax2.plot(f, F_f.imag)
	ax2.set_xlim(-0.5, 0.5)
	ax2.set_ylim(-10, 10)
	plt.show()
