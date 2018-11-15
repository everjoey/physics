#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg

plt.ion()

N = 1001
dx = 0.01
dt = 0.01



V = np.zeros([N-2, N-2])

#V[(N-1)/2][(N-1)/2] = 10
#V[(N)/2][(N)/2] = 10
#V[(N+1)/2][(N+1)/2] = 10

psi = np.zeros([2,N], dtype='complex')

x = dx*np.linspace(0,N-1,N)

psi[0] = np.sin(0.2*np.pi*x)
#psi[0] = x*x*x*(10-x)

A = np.sqrt((psi[0]**2*dx).sum())
psi[0] = psi[0]/A

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,psi[0].real)
ax.set_ylim(-1,1)

A = np.zeros([N,N])

for i in range(1,N-1):
	A[i][i-1:i+2] = np.array([1,-2,1])

r = 1j*dt/(2*dx**2)

AA = -r*A[1:N-1][:,1:N-1]+np.identity(N-2)+V

AAA = linalg.inv(AA)

for i in range(5*N):
	psi[1][1:-1] = np.dot(AAA, psi[0][1:-1])
#	psi[1][1:-1]=r*(psi[0][:-2]-2*psi[0][1:-1]+psi[0][2:])+psi[0][1:-1]
	if i%50 == 0:
		ax.lines[0].set_ydata(psi[1].real)
#		ax1.lines[1].set_ydata(psi1[2].imag)
		plt.draw()

	psi[0]=psi[1]
plt.ioff()
plt.show()
