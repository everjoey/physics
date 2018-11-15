#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate

def f(y,x):
	return np.sqrt(2/y-1)

#x = np.linspace(0,1,1001)
#y = integrate.odeint(f, 1, x)
#print(y)

fig = plt.figure()
ax = fig.add_subplot(111)

t = np.linspace(0,10,1001)
x = t - 1/2*np.sin(2*t)
y = np.sin(t)**2

ax.plot(x,y)



plt.show()

