#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

plt.ion()

t = np.linspace(0,100,1000)

theta_int = np.array([0,0])
x_int = np.sin(np.pi/3)
y_int = np.cos(np.pi/3)

g = 0.1
a = 0.4
b = 1
w= 0.2
def f(theta,t):
	return np.array([theta[1], w**2*a/b*np.cos(theta[0]-w*t)-g/b*np.sin(theta[0])])

#print(f(theta_int))


theta = integrate.odeint(f,theta_int,t)
print(theta[:,0])

#x = np.sin(theta[:,0])
#y = np.cos(theta[:,0])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(-1,1)
ax.set_ylim(-2,1)
ax.set_aspect('equal')

ax.plot(a, 0, 'o', a+np.sin(np.pi/3), -b*np.cos(np.pi/3), 'o', np.linspace(a,a+np.sin(np.pi/3),100), np.linspace(0,-b*np.cos(np.pi/3),100))
j=0

for i in range(1000):
	x0 = a*np.cos(w*i)
	y0 = a*np.sin(w*i)
	x1 = a*np.cos(w*i)+b*np.sin(theta[:,0][i])
	y1 = a*np.sin(w*i)-b*np.cos(theta[:,0][i])
	ax.lines[0].set_xdata(x0)
	ax.lines[0].set_ydata(y0)
	ax.lines[1].set_xdata(x1)
	ax.lines[1].set_ydata(y1)
	ax.lines[2].set_xdata(np.linspace(x0,x1,100))
	ax.lines[2].set_ydata(np.linspace(y0,y1,100))
	if j%2==0:
		plt.draw()
	j=j+1

plt.ioff()
plt.show()

