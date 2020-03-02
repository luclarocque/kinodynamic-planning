# sampling functions

import numpy as np
import random

# uniformly sample n times in the rectangular prism with bottom-left corner at
#   origin, tuple of form (x,y,z). Returns a list containing the sample nodes.
def sample(n, origin, width, depth, height, COST, ax=None, zero_vel=False):
    from fmt_quad import Node#, cost, xinit
    L = set([])
    for i in range(n):
        x = origin[0] + random.random()*width
        y = origin[1] + random.random()*depth
        z = origin[2] + random.random()*height
        if zero_vel:
            vx = 0
            vy = 0
            vz = 0
        else:
            vx = (2*random.random() - 1)*5  # values between (-5, 5)
            vy = (2*random.random() - 1)*5
            vz = (2*random.random() - 1)*5
        v = Node((x, y, z, vx, vy, vz))
        # v.cost = cost(xinit, v, COST)
        L |= set([v])
    if ax != None:
        arrL = np.asarray(list(map(lambda x: x.state, list(L))))
        ax.scatter(arrL[:, 0], arrL[:, 1], arrL[:, 2])  # plot points
    return L


# selects Npair (<= n*(n-1)) states with replacement from set V and
#   returns a set of pairs of states (a,b) for use in building
#   the cost-lookup table/roadmap
def sample_data(V, Npair):
    lstV = list(V)
    S = set([])
    num_nodes = len(lstV)
    for i in range(Npair):
        randind1 = np.random.randint(num_nodes)
        randind2 = np.random.randint(num_nodes)
        S |= set([(lstV[randind1].state, lstV[randind2].state)])
    return S