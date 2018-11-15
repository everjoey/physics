#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def rect(x,a,w):
	if x>-w/2+a and x<w/2+a:
		return 1
	elif x==-w/2+a or x==w/2+a:
		return 1/2
	else:
		return 0
rect_v = np.vectorize(rect)

def Gaussian(x,t,sigma):
    """  A Gaussian curve.
        x = Variable
        t = time shift
        sigma = standard deviation      """
    return np.exp(-(x-t)**2/(2*sigma**2))



plt.ion()

N=2001
dx = 1
dt = 0.5
c = 1
#x = dx*np.linspace(-(N-1)/2,(N-1)/2,N)
x = dx*np.linspace(0,N-1,N)
f = np.fft.fftfreq(len(x))
print(f)
print(max(f), min(f))
u1 = np.zeros([3,N], dtype='complex')
u2 = np.zeros([3,N], dtype='complex')

u1[0] = Gaussian(x,50,10)#*np.exp(1j*np.pi*0.05*x)
u2[0] = Gaussian(x,50,10)#*np.exp(1j*np.pi*0.05*x)

fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(221)
ax1.plot(x,u1[0].real, x,u1[0].imag)
ax1.set_ylim(-1,1)


u2[1] = np.fft.fft(u2[0])
ax2 = fig.add_subplot(222)
ax2.plot(f,u2[1].real,f,u2[1].imag, f, np.sqrt(np.pi*200)*np.cos(2*np.pi*f*50)*np.exp(-200*np.pi**2*f**2), f, -np.sqrt(np.pi*200)*np.sin(2*np.pi*f*50)*np.exp(-200*np.pi**2*f**2))

u2[2] = np.fft.ifft(u2[1])
ax3 = fig.add_subplot(223)
ax3.plot(x,u2[2].real,x,u2[2].imag)
ax3.set_ylim(-1,1)
plt.draw()

u1[1] = Gaussian(x-dt,50,10)
#u1[1][1:-1] = (u1[0][:-2] + u1[0][2:])/2

r = dt**2/(dx**2)
print(r)

for i in range(2*N):
	u1[2][1:-1]=2*(1-r)*u1[1][1:-1]+r*u1[1][:-2]+r*u1[1][2:]-u1[0][1:-1]
	u2[2] = np.fft.ifft(u2[1]*np.exp(-1j*2*np.pi*f*c*dt*i))
	if i%50 == 0:
		ax1.lines[0].set_ydata(u1[2].real)
		ax1.lines[1].set_ydata(u1[2].imag)
		ax3.lines[0].set_ydata(u2[2].real)
		ax3.lines[1].set_ydata(u2[2].imag)
		plt.draw()

	u1[0]=u1[1]
	u1[1]=u1[2]

plt.ioff()
plt.show()
