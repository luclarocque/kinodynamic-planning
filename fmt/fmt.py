import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from line_intersect import collision_free
import timing

######


class Node:
    state = (0., 0.)
    cost = 0
    parent = None
    children = set([])

    def __init__(self, statex):
        self.state = statex


class Edge:
    source = None
    target = None

    def __init__(self, srcnode, tarnode):
        self.source = srcnode
        self.target = tarnode

# class Graph:
#     V = []
#     E = []
#     def __init__(self, v, e):
#         self.V = v
#         self.E = e

######

xinit = Node((4.2, 9.))

Xgoal_pos = (0., 0.)
Xgoal_height = 1.
Xgoal_width = 1.

n = 490  # number to sample
n_goal = 10  # number to sample from Xgoal

Jth = 5.
######

# set the size of the workspace

XDIM = 10.
YDIM = 10.
# ZDIM = 10.

dim = 2  # total number of dimensions
posdim = 2  # number of spatial dimensions (e.g., 2D, 3D)

# OBS = []
OBS = [[1.5, 1.5, 1.5, 6.],
       [5., 8., 4., 2.],
       [3.5, 3., 1, 5.5],
       [3.5, 0.5, 6., 1.5],
       [6., 3.5, 2.5, 3.]]


# uniformly sample n times in the rectangle with bottom-left corner at
#   origin, a pair (x,y). Returns a list containing the sample nodes.

def sample(n, origin, width, height):
    L = set([])
    for i in range(n):
        x = origin[0] + random.random()*width
        y = origin[1] + random.random()*height
        v = Node((x, y))
        v.cost = cost(xinit, v)
        L |= set([v])
    # arrL = np.asarray(list(map(lambda x: x.state, list(L))))
    # plt.scatter(arrL[:, 0], arrL[:, 1], zorder=2, marker='.')  # plot points
    return L


# cost from node n1 to n2. (In this case, just distance)
def cost(n1, n2):
    total = 0
    for i in range(dim):
        cur = (n1.state[i]-n2.state[i])
        cur = cur*cur
        total += cur
    return total


# finds the nodes of S that are within distance/cost J of node x.
#   Note: this should be the forward-reachable set. TODO Use SVM.
def near(x, S, J):
    aset = set([])
    for s in S:
        if cost(x, s) < J*J:  # square J since cost is actually dist squared
            aset |= set([s])
    return aset


def nearest(S, newnode):
    nn = S[0]
    for p in S:
        if cost(p, newnode) < cost(nn, newnode):
            nn = p
    return nn


# plot solution path from endnode to origin, and return waypoints as list
def draw_path(endnode, ax):
    deq = deque()
    source = endnode.parent
    target = endnode
    deq.appendleft(target)
    while target != xinit:
        deq.appendleft(source)
        if source is None:
            break
        xs = (source.state[0], target.state[0])
        ys = (source.state[1], target.state[1])
        ax.plot(xs, ys, lw=3, color='b', zorder=1)

        target = source
        source = source.parent
    return list(deq)


# Given node x and goal region variables, determine if x is in the goal region.
def in_goal(x, Xgoal_pos, Xgoal_height, Xgoal_width):
    return (Xgoal_pos[0] < x.state[0] < Xgoal_pos[0] + Xgoal_width and
            Xgoal_pos[1] < x.state[1] < Xgoal_pos[1] + Xgoal_height)


def main():

    # setup environment
    ax = plt.axes(xlim=(0., 10.), ylim=(0., 10.))
    plt.scatter(xinit.state[0], xinit.state[1],
                zorder=2, color='g', marker='H', s=100)

    for o in OBS:
        new_obs = plt.Rectangle((o[0], o[1]), o[2], o[3], fc='r')
        plt.gca().add_patch(new_obs)

    Xgoal_rect = plt.Rectangle(Xgoal_pos, Xgoal_width, Xgoal_height, fc='g')
    plt.gca().add_patch(Xgoal_rect)

    ######
    # initialize graph

    V = set([xinit])
    E = set([])

    V |= sample(n_goal, Xgoal_pos, Xgoal_width, Xgoal_height)
    V |= sample(n, (0., 0.), XDIM, YDIM)

    H = set([xinit])

    W = V - set([xinit])  # copy of V setminus {xinit}

    z = xinit

    ######

    while not in_goal(z, Xgoal_pos, Xgoal_height, Xgoal_width):
        N_zout = near(z, V - set([z]), Jth)
        Xnear = N_zout & W  # Intersection
        for x in Xnear:
            N_xin = near(x, V - set([x]), Jth)
            Ynear = N_xin & H
            if len(Ynear) == 0:
                continue
            ymin = random.choice(tuple(Ynear))
            ymin_x_cost = ymin.cost + cost(ymin, x)
            for y in Ynear:
                y_x_cost = y.cost + cost(y, x)
                if y_x_cost < ymin_x_cost:
                    ymin = y
                    ymin_x_cost = y_x_cost
            if collision_free(ymin, x, OBS):
                ymin.children |= set([x])
                x.parent = ymin
                x.cost = ymin_x_cost
                E |= set([Edge(ymin, x)])
                H |= set([x])
                W -= set([x])
                xs = (ymin.state[0], x.state[0])
                ys = (ymin.state[1], x.state[1])
                line = plt.Line2D(xs, ys, lw=1, c=[0.2, 0.2, 0.2],
                                  marker='o', ms=1, mfc='k')
                plt.gca().add_line(line)
        H -= set([z])
        if len(H) == 0:
            print("Failure")
            return -1
        z = random.choice(tuple(H))
        # find argmin y in H of cost-to-come
        for h in H:
            if h.cost < z.cost:
                z = h

    waypoints = draw_path(z, ax)
    plt.axis('scaled')
    plt.show('hold')
    return z


# def test():
#     plt.axes(xlim=(0., 10.), ylim=(0., 10.))

#     obs = [[2, 3, 6, 1],
#            [3, 4.5, 1, 4]]
#     for o in obs:
#         new_obs = plt.Rectangle((o[0], o[1]), o[2], o[3], fc='r')
#         plt.gca().add_patch(new_obs)

#     x0 = Node((1, 1))
#     x1 = Node((3, 1))
#     x2 = Node((4, 4))
#     x3 = Node((7, 6))
#     x4 = Node((6, 0))

#     L = [x0, x1, x2, x3, x4]
#     arrL = np.asarray(list(map(lambda x: x.state, list(L))))
#     plt.scatter(arrL[:, 0], arrL[:, 1], zorder=2)

#     pair1 = (x0.state[0], x1.state[0])
#     pair2 = (x0.state[1], x1.state[1])
#     line = plt.Line2D(pair1, pair2, lw=1)
#     plt.gca().add_line(line)

#     plt.axis('scaled')
#     plt.show('hold')

if __name__ == '__main__':
    main()
