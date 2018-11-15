#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import axes3d
import time
import scipy.signal as sig

plot_mode = 1

def rect(x,a,w):
	if x>-w/2+a and x<w/2+a:
		return 1
	elif x==-w/2+a or x==w/2+a:
		return 1/2
	else:
		return 0

rect_v = np.vectorize(rect)

plt.ion()
N=101
dx = 0.01
dy = 0.01
dt = 0.005

x = dx*np.linspace(0,N-1,N)
y = dy*np.linspace(0,N-1,N)
x,y = np.meshgrid(x,y)

r = (dt/dx)**2

u = np.zeros([3,N,N])
u[0] = np.sin(1*np.pi*x)*np.sin(1*np.pi*y)
u[0] = np.sin(1*np.pi*x)*np.sin(1*np.pi*y)+np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
#u[0][40:60][:,40:60] = np.zeros((20,20))+1
#u[0] = sig.triang(N)*np.sin(np.pi*y)

if plot_mode == 1:
	fig = plt.figure()
	ax = fig.add_axes([0.1,0.1,0.8,0.8], projection='3d')
	ax.set_zlim(-1,1)
	ax.plot_surface(x,y,u[0],cmap=cm.jet, vmin=-1, vmax=1)
else:
	fig = plt.figure()
	ax = fig.add_axes([0.1,0.1,0.8,0.8])
	axi = ax.imshow(u[0], cmap=cm.jet, vmin=-1, vmax=1, extent=(0,1,1,0))
	axc = ax.contour(x,y,u[0],colors='k',levels=np.arange(-1,1.2,0.1))

plt.draw()



u[1][1:-1][:,1:-1] = (u[0][:-2][:,1:-1] + u[0][2:][:,1:-1] + u[0][1:-1][:,:-2] + u[0][1:-1][:,2:] - 2*u[0][1:-1][:,1:-1])/2

if plot_mode == 1:
	ax.collections=[]
	ax.plot_surface(x,y,u[1],cmap=cm.jet,vmin=-1,vmax=1)
else:
	axi.set_data(u[1])
	ax.collections=[]
	ax.contour(x,y,u[1],colors='k',levels=np.arange(-1,1.2,0.1))

plt.draw()



for i in range(1000):
	u[2][1:-1][:,1:-1] =r*(\
	u[1][:-2][:,1:-1]+u[1][2:][:,1:-1]+\
	u[1][1:-1][:,:-2]+u[1][1:-1][:,2:]-\
	4*u[1][1:-1][:,1:-1])-\
	u[0][1:-1][:,1:-1]+\
	2*u[1][1:-1][:,1:-1]

	if plot_mode == 1:
		ax.collections=[]
		ax.plot_surface(x,y,u[2],cmap=cm.jet,vmin=-1,vmax=1)
	else:
		axi.set_data(u[2])
		ax.collections=[]
		ax.contour(x,y,u[2],colors='k',levels=np.arange(-1,1.2,0.1))

	plt.draw()
	print(u[2])
	u[0]=u[1]
	u[1]=u[2]

plt.ioff()
plt.show()


