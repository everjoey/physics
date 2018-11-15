#!/usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from scipy import integrate
from scipy import optimize
from scipy import special

from numpy.polynomial import polynomial
from numpy.polynomial import legendre
from numpy.polynomial import hermite



class LinearDifferentialEquation(object):
	def __init__(self, coefficients, source):
		self.coefficients = coefficients
		self.source = source

	def solve(self, init, x):
		order = len(self.coefficients)-1
		self.x = x
		A = np.zeros((order, order))
		b = np.zeros(order)

		def f(y, x):
			for i in range(order):
				for j in range(order):
					if i+1 == j:
						A[i][j] = 1
					elif i == order-1:
						A[i][j] = -self.coefficients[j](x)/self.coefficients[-1](x)
					else:
						A[i][j] = 0
			b[-1] = self.source(x)

			return np.dot(A, y) + b 
		self.y = integrate.odeint(f, init, x)
		return self.y

	def plot(self, order=slice(0,2)):
		fig = plt.figure()
		ax = fig.add_axes([0.1,0.1,0.8,0.8])
		l = ax.plot(self.x, self.y[:,order])

	def f(self):
		print('super')

class CauchyEulerEquation(LinearDifferentialEquation):
	def solve(self, init, x):
		'''
		def f(m):
			sum = 0
			for i in range(len(self.coefficients)):
				factorial = 1
				for j in range(i):
					if i == 0:
						factorial = 1
					else:
						factorial = factorial*(m-j)
				sum = sum + self.coefficients[i]*factorial
			return sum
		print(f(6))
		'''


class LaguerreEquation(LinearDifferentialEquation):
	def __init__(self, function, degree, order):
		self.function = function
		self.degree = degree
		self.order = order
#		coefficients[0] = 
#		coefficients[1] = (1-function)*function.deriv(1)**2-function*function.deriv(2)
#		coefficients[2] = function*function.deriv(1)


class LegendreEquation(LinearDifferentialEquation):
	def __init__(self, function, degree, order):

		if isinstance(degree, int) and degree>=0:
			self.degree = degree
		else:
			raise ValueError('The degree of the associated Legendre functions must be non-negtive intenger!')

		if isinstance(order, int) and -degree<=order<=degree:
			self.order = order
		else:
			raise ValueError
	
		if isinstance(function, polynomial.Polynomial):
			self.function = function
		else:
			raise ValueError

		coefficients = [0 for i in range(3)]
		coefficients[0] = function.deriv(1)**3*(degree*(degree+1)*(1-function**2)-order**2)
		coefficients[1] = (-2*function*function.deriv(1)**2-function.deriv(2)*(1-function**2))*(1-function**2)
		coefficients[2] = function.deriv(1)*(1-function**2)**2
		self.coefficients = coefficients
		self.source = np.polynomial.polynomial.Polynomial([0])

	def solve(self , init , x):
		A = np.zeros([2,2])
		A[0][0] = special.lpmn(self.order, self.degree, self.function(x[0]))[0][self.order][self.degree]
		A[0][1] = special.lqmn(self.order, self.degree, self.function(x[0]))[0][self.order][self.degree]
		A[1][0] = self.function.deriv(1)(x[0])*special.lpmn(self.order, self.degree, self.function(x[0]))[1][self.order][self.degree]
		A[1][1] = self.function.deriv(1)(x[0])*special.lqmn(self.order, self.degree, self.function(x[0]))[1][self.order][self.degree]
		print(A)
		c = np.linalg.solve(A, init)
		self.c = c
		y = np.zeros((len(x), 2))
		for i in range(len(x)):
			for j in range(2):
				if j == 0:
					y[i][j] = \
					c[0]*special.lpmn(self.order, self.degree, self.function(x[i]))[j][self.order][self.degree] + \
					c[1]*special.lqmn(self.order, self.degree, self.function(x[i]))[j][self.order][self.degree]

				elif j == 1:
					y[i][j] = \
					c[0]*self.function.deriv(1)(x[i])*special.lpmn(self.order, self.degree, self.function(x[i]))[j][self.order][self.degree] + \
					c[1]*self.function.deriv(1)(x[i])*special.lqmn(self.order, self.degree, self.function(x[i]))[j][self.order][self.degree]
		self.x = x
		self.y = y
		return y




class RadialSphericalHelmholtzEquation(LinearDifferentialEquation):
	def __init__(self, function, l):
		self.function = function
		self.l = l

		coefficients = [0 for i in range(3)]
		coefficients[0] = function.deriv(1)**3*(function**2-l*(l+1))
		coefficients[1] = 2*function*function.deriv(1)**2-function.deriv(2)*function**2
		coefficients[2] = function.deriv(1)*function**2
		self.coefficients = coefficients
		self.source = np.polynomial.polynomial.Polynomial([0])

	def solve(self, init, x):
		A = np.zeros((2,2))
		A[0][0] = special.sph_jn(self.l,self.function(x[0]))[0][self.l]
		A[0][1] = special.sph_yn(self.l,self.function(x[0]))[0][self.l]
		A[1][0] = self.function.deriv(1)(x[0])*special.sph_jn(self.l,self.function(x[0]))[1][self.l]
		A[1][1] = self.function.deriv(1)(x[0])*special.sph_yn(self.l,self.function(x[0]))[1][self.l]

		c = np.linalg.solve(A, init)

		y = np.zeros((len(x), 2))
		for i in range(len(x)):
			for j in range(2):
				if j == 0:
					y[i][j] = \
					c[0]*special.sph_jn(self.l,self.function(x[i]))[j][self.l] + \
					c[1]*special.sph_yn(self.l,self.function(x[i]))[j][self.l]

				elif j == 1:
					y[i][j] = \
					c[0]*self.function.deriv(1)(x[i])*special.sph_jn(self.l,self.function(x[i]))[j][self.l] + \
					c[1]*self.function.deriv(1)(x[i])*special.sph_yn(self.l,self.function(x[i]))[j][self.l]
		self.x = x
		self.y = y
		return y

	def f(self):
		print('low')

def plot(x,y):
	fig = plt.figure()
	ax = fig.add_axes([0.1,0.1,0.8,0.8])
	ax.set_ylim(-1,1)
	l = ax.plot(x, y)

if __name__ == '__main__':
	pass
#	t=np.linspace(-10,10,1000)
#	y = np.tanh(t)
#	plot(t,y)
#	plt.show()
