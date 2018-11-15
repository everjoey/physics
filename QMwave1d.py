#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import time

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
dt = 0.1
c = 1
#x = dx*np.linspace(-(N-1)/2,(N-1)/2,N)
x = dx*np.linspace(0,N-1,N)
f = np.fft.fftfreq(len(x))

psi1 = np.zeros([3,N], dtype='complex')
psi2 = np.zeros([3,N], dtype='complex')
#u[0] = rect_v(x,0.5,0.2)
#u[0] = sig.gaussian(N, 100)#*np.exp(1j*x*20)
#u[0] = sig.gausspulse(x, bw=5, fc=5, retquad=False, retenv=False)

psi1[0] = Gaussian(x,1000,10)#*np.exp(1j*np.pi*0.05*x)
psi2[0] = Gaussian(x,1000,10)#*np.exp(1j*np.pi*0.05*x)
p1 = psi1[0].real**2+psi1[0].imag**2

fig = plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(221)
ax1.plot(x,psi1[0].real,x,psi1[0].imag,x,p1)
ax1.set_ylim(-1,1)


psi2[1] = np.fft.fft(psi2[0])
ax2 = fig.add_subplot(222)
ax2.plot(\
f,psi2[1].real,\
f,psi2[1].imag,\
#f,np.sqrt(np.pi*200)*np.cos(2*np.pi*f*1000)*np.exp(-200*np.pi**2*f**2),\
#f,-np.sqrt(np.pi*200)*np.sin(2*np.pi*f*1000)*np.exp(-200*np.pi**2*f**2)\
)
print(f)
psi2[2] = np.fft.ifft(psi2[1])
ax3 = fig.add_subplot(223)
ax3.plot(x,psi2[2].real,x,psi2[2].imag)
ax3.set_ylim(-1,1)

plt.draw()

#psi1[1] = Gaussian(x-dt,50,10)#initial condition
#psi1[1][1:-1] = (psi1[0][:-2] + psi1[0][2:])/2
psi1[1] = psi1[0]
p1 = psi1[1].real**2+psi1[1].imag**2

r = 1j*dt/dx**2
print(r)

for i in range(5*N):
	psi1[2][1:-1]=r*(psi1[1][:-2]-2*psi1[1][1:-1]+psi1[1][2:])+psi1[0][1:-1]
	psi2[2] = np.fft.ifft(psi2[1]*np.exp(-1j*(2*np.pi*f)**2*0.5*dt*i))

	p1 = psi1[2].real**2+psi1[2].imag**2

	if i%50 == 0:
		ax1.lines[0].set_ydata(psi1[2].real)
		ax1.lines[1].set_ydata(psi1[2].imag)
		ax1.lines[2].set_ydata(p1)
		ax3.lines[0].set_ydata(psi2[2].real)
		ax3.lines[1].set_ydata(psi2[2].imag)
		plt.draw()

	psi1[0]=psi1[1]
	psi1[1]=psi1[2]



'''
for i in range(8*N):
#	u[2][1:-1] = u[1][:-2]+u[1][2:]-u[0][1:-1]

	u[2][1:-1]=\
	2*(1-r)*u[1][1:-1]+\
	r*u[1][:-2]+\
	r*u[1][2:]-\
	u[0][1:-1]


#	u[2][1:-1]=\
#	-2*r*u[1][1:-1]+\
#	r*u[1][:-2]+\
#	r*u[1][2:]+\
#	u[0][1:-1]

	if i%50 == 0:
		l1.set_ydata(u[2].real)
		l2.set_ydata(u[2].imag)
		plt.draw()

	u[0]=u[1]
	u[1]=u[2]
'''
plt.ioff()
plt.show()
