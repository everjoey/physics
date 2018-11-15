#!/usr/bin/env python3
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

n=2
l=1
m=0
a0=1
x = np.linspace(-n**3,n**3,100)
z = np.linspace(-n**3,n**3,100)
x, z = np.meshgrid(x,z)
'''
Radial Equation
'''
r = np.sqrt(np.square(x)+np.square(z))

R1 = np.exp(-r/n/a0)
R2 = (2*r/n/a0)**l
R3 = special.genlaguerre(n-l-1,2*l+1)(2*r/n/a0)

R = np.sqrt((2/n/a0)**3*np.math.factorial(n-l-1)/(2*n*np.math.factorial(n+l)))*R1*R2*R3
Y = special.sph_harm(m,l,0,np.arccos(z/r)).real
#print(R)
#print(Y)
f = Y*R

print(f)


#f = 1/4/np.sqrt(2)/np.sqrt(np.pi)*r*np.exp(-r/2)*z/r
print(f)
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
axi = ax.imshow(f, extent=[-1, 1, -1, 1])
fig.colorbar(axi)

plt.show()



