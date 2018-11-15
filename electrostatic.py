#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import epsilon_0

epsilon = 1#epsilon_0
N = 1000
dr = 1
r = dr*np.linspace(0,N-1,N)
Er = np.zeros(N)

def rho(r):
	if r < 250:
		return epsilon
	else:
		return 0

for i in range(N-1):
	Er[i+1] = dr*rho(r[i+1])/epsilon+(1-2*dr/r[i+1])*Er[i]

fig = plt.figure()
ax = fig.add_subplot(111)
l1 = ax.plot(r, Er)
plt.show()
