#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

a=1
e = 1.5
c=a*e
b=np.sqrt(np.abs(a**2-c**2))
l = b**2/a

theta = np.linspace(np.pi*0.7,np.pi*1.3, 361)

r1 = -l/(1+e*np.cos(theta))
#r2 = l/(1+e*np.cos(theta))



ax.plot(theta, r1)
ax.set_rmax(10)
print(ax.lines)
plt.show()
