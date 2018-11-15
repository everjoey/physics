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
dt = 0.4

#x = dx*np.linspace(-(N-1)/2,(N-1)/2,N)
x = dx*np.linspace(0,N-1,N)
f = np.fft.fftfreq(len(x))

u = np.zeros([3,N], dtype='complex')
#u[0] = np.sin(np.pi*x)
#u[0] = rect_v(x,0.5,0.2)
#u[0] = sig.gaussian(N, 100)#*np.exp(1j*x*20)
#u[0] = sig.gausspulse(x, bw=5, fc=5, retquad=False, retenv=False)
#u[0] = np.cos(5*np.pi*x)
#u[1] = np.cos(5*np.pi*(x-dt))
#u[1][1:-1] = (u[0][:-2] + u[0][2:])/2

u[0] = Gaussian(x,50,10)#*np.exp(1j*np.pi*0.05*x)



#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax1.plot(x,u[0].real, x,u[0].imag)
#ax1.set_ylim(-1,1)

U = np.fft.fft(u[0])
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.plot(f,U.real,f,U.imag, f, np.sqrt(np.pi*200)*np.cos(2*np.pi*f*50)*np.exp(-200*np.pi**2*f**2), f, -np.sqrt(np.pi*200)*np.sin(2*np.pi*f*50)*np.exp(-200*np.pi**2*f**2))


u[2] = np.fft.ifft(U)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(x,u[2].real,x,u[2].imag)
ax3.set_ylim(-1,1)
plt.draw()

u[1] = Gaussian(x+dt,500,50)

#r = (dt/dx)**2
r = dt**2/(dx**2)
print(r)

for i in range(8*N):
	u[1] = np.fft.ifft(U*np.exp(-1j*2*np.pi*f*10*dt*i))
	if i%50 == 0:
		ax3.lines[0].set_ydata(u[1].real)
		ax3.lines[1].set_ydata(u[1].imag)
		plt.draw()




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
