#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

plt.ion()

t = np.linspace(0,100,1000)

theta_int = np.array([np.pi/3,0])
x_int = np.sin(np.pi/3)
y_int = np.cos(np.pi/3)


def f(theta,t):
	return np.array([theta[1], -0.1*np.sin(theta[0])])

#print(f(theta_int))


theta = integrate.odeint(f,theta_int,t)
print(theta[:,0])

#x = np.sin(theta[:,0])
#y = np.cos(theta[:,0])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-1,1)
ax.set_ylim(2,0)



ax.plot(x_int, y_int, 'o', np.linspace(0,x_int,100), np.linspace(0,y_int,100))
j=0
for i in theta[:,0]:
	x = np.sin(i)
	y = np.cos(i)
	ax.lines[0].set_xdata(x)
	ax.lines[0].set_ydata(y)
	ax.lines[1].set_xdata(np.linspace(0,x,100))
	ax.lines[1].set_ydata(np.linspace(0,y,100))
	if j%2==0:
		plt.draw()
	j=j+1

plt.ioff()
plt.show()

