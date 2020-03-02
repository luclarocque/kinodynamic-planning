import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import save_obj
from numpy.polynomial.polynomial import polyval
import sys

NUM = 50

sys.setrecursionlimit(200000)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X')
ax.set_xlim3d(0, 10)
ax.set_ylabel('Y')
ax.set_ylim3d(0, 10)
ax.set_zlabel('Z')
ax.set_zlim3d(0, 10)

P = save_obj.load_object('P.pkl')

M = 3
N = 9
T = [1.7608794149125462, 1.5615551748592407, 1.1402720224000513]

for m in range(M):
    tt = np.linspace(0, T[m], num=NUM)
    xs = polyval(tt, P['x'][m*(N+1):(m+1)*(N+1)]).reshape(NUM,)
    ys = polyval(tt, P['y'][m*(N+1):(m+1)*(N+1)]).reshape(NUM,)
    zs = polyval(tt, P['z'][m*(N+1):(m+1)*(N+1)]).reshape(NUM,)
    yaws = polyval(tt, P['yaw'][m*(N+1):(m+1)*(N+1)]).reshape(NUM,)
    ax.plot(xs, ys, zs, lw=2, color='g', zorder=2)


plt.show()
