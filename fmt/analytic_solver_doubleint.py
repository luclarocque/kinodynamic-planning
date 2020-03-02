# Analytical solution to: Double Integrator OBVP

import numpy as np
import scipy.linalg
from functools import reduce

# Cost:  int_0^\tau (1 + u.T Ru u) dt

g = -9.8
wr = 1.  # control penalty weight
Ru = wr * np.eye(3)  # control penalty matrix

####
A = np.zeros([6,6])
A[0:3,3:] = np.eye(3)

B = np.zeros([6,3])
B[3:,:] = np.eye(3)

c = np.zeros(6)
c[-1] = g

# x0 = np.zeros(6)  # initial state
# x1 = np.ones(6)  # goal state

# tau = 0.8668  # initial guess for optimal final time


####
def iterdot(*args):
    return reduce(np.dot, args)
####


def G(t):
    return 1./wr * np.array([[t**3/3., 0., 0., t**2/2., 0., 0.],
                             [0., t**3/3., 0., 0., t**2/2., 0.],
                             [0., 0., t**3/3., 0., 0., t**2/2.],
                             [t**2/2., 0., 0., t, 0., 0.],
                             [0., t**2/2., 0., 0., t, 0.],
                             [0., 0., t**2/2., 0., 0., t]])


def xbar(x0, t):
    expAt = scipy.linalg.expm(A*t)  # matrix exponential
    return np.dot(expAt, x0) + g*np.array([0., 0., t**2/2., 0., 0., t])


def d(x0, x1, t):
    return np.dot( np.linalg.inv(G(t)), (x1 - xbar(x0, t)) )


# optimal cost from x0 to x1 with fixed final time t
def Jstar(x0, x1, t):
    xb = xbar(x0, t)
    return t + iterdot( (x1 - xb), np.linalg.inv(G(t)), (x1 - xb) )


####


def u(x0, x1, t, tau):
    expAt = scipy.linalg.expm(A.T*(tau - t)) # matrix exponential
    return iterdot( np.linalg.inv(Ru), B.T, expAt, d(x0, x1, tau) )


def x(x0, x1, t, tau):
    expAt = scipy.linalg.expm(A.T*(tau - t)) # matrix exponential
    return xbar(x0, t) + iterdot( G(t), expAt, d(x0, x1, tau) )


####


# compute numerical derivative of fcn evaluated at t with small h
def deriv(fcn, t, h):
    return (fcn(t+h) - fcn(t))/h


# minimize a convex scalar function fcn using initial interval [a,b]
#   and tolerance tol
def minimize(fcn, interval, tol):
    a = interval[0]
    b = interval[1]
    c = (a + b)/2.
    deriv_c = deriv(fcn, c, tol)

    if b-a < tol:
        return c
    elif deriv(fcn, a, tol) * deriv_c < 0:  # check if derivative has changed sign
        return minimize(fcn, [a,c], tol)
    elif deriv(fcn, b, tol) * deriv_c < 0:
        return minimize(fcn, [c,b], tol)
    else:
        return c


# find the optimal fixed end-time given states x0, x1
def find_tau(x0, x1):
    return minimize(lambda t: Jstar(x0, x1, t), [0.001, 10.], 0.000000001)

# tt = minimize(lambda t: Jstar(x0, x1, t), [0.0001, 100], 0.000000001 )
# print(tt)