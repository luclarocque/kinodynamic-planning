import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

Tprop = 1.0
Xvec = np.array([10.0, 10.0, 0.0, 0.0])
Ux = [-5.0, 5.0]
Uy = [0.0, 20.0]

u = np.array([0,0])
t = 0

def distxy(p1,p2):
    return (p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])

def fcn(X,t,u):
    return np.array([X[2], X[3], u[0], -9.8 + u[1]])

def MonteCarlo_Prop(x, Tprop):
    h = 0.01
    T = random.random()*Tprop
    t = np.linspace(0,T, num=T/h)
    u = np.array([random.uniform(Ux[0],Ux[1]), random.uniform(Uy[0],Uy[1])])
    xnew = odeint(fcn, x, t, (u,))
    runcost = 0
    for i in xrange(len(xnew)-1):
        runcost += distxy(xnew[i],xnew[i+1])
    return [t,xnew,u,runcost]

t,x,u,cost = MonteCarlo_Prop(Xvec, Tprop)
print "control: u=[%g,%g]" % (u[0],u[1])

print x[-1]
plt.plot(x[:,0],x[:,1])
plt.show()

