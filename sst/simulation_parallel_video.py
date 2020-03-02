#!/usr/bin/env python

# Set multiple initial states (one for each ) from which to build trees

import sst_doubleint_parallel as sst
from sst_doubleint_parallel import MultiGraph, Graph, Node, Edge
from lqr import *
from save_obj import load_object
import numpy as np
import pygame
from scipy import integrate
from pygame.locals import *
from time import sleep
import os
import colorsys

h = sst.h  # numercial integration timestep
num_props = sst.num_props

# Initialization
MG = sst.MGinit
MG.graphs = load_object('MGgraphs_doubleint_vid2.pkl')
endnodes = load_object('endnodes_doubleint_vid2.pkl')
# MG.graphs = load_object('MGdoubleint2.pkl')
# endnodes = load_object('endnodes_doubleint2.pkl')


# Obtain list of edges of the candidate trajectory given endnode
def construct_edgelist(endnode, G, xinit):
    edgelst = []
    timecost = 0
    controlcost = 0
    curnode = endnode
    while not(np.array_equal(curnode.state, xinit.state)):
        for e in G.E:
            if np.array_equal(e.target.state, curnode.state):
                u = e.control.reshape(sst.udim, 1)
                edgelst.append(e)
                timecost += e.time
                controlcost += np.linalg.multi_dot([u.T, R, u])
                curnode = e.source
                break
    cost = timecost + controlcost
    edgelst.reverse()
    return edgelst, cost


# init_colour (rgb) transformed to different colour
def animate_colour(init_colour):
    colour = list(colorsys.rgb_to_hsv(*init_colour))
    colour[0] += np.pi/60.
    colour[1] = 0.7
    colour[2] = 0.95
    colour = colorsys.hsv_to_rgb(*colour)
    return np.array(colour)*255


# simulate trajectory starting at x and along the edges of edgelst
def simulate(x, edgelst):
    global iternum
    colour = np.array([0.4, 0.95, 0.95])
    for e in edgelst:
        colour = animate_colour(colour/255.)
        T = e.time
        t = np.linspace(0, T, num=int(T/h) + 1)
        tmp = x
        for now in range(len(t)-1):
            tt = np.linspace(t[now], t[now]+h, num=2) # int forward one step
            err = np.array(x.state) - np.array(e.traj[now+1])
            u = e.control - np.array(np.dot(K, err))
            # for ind in xrange(sst.udim):
            #     if abs(u[ind]) > sst.Umax:
            #         u[ind] = u[ind]/abs(u[ind]) * sst.Umax
            # print("u: {}".format(u))
            x = integrate.odeint(fcn, x.state, tt, (u,))
            x = sst.Node(x[-1])  # proceed from latest integrated state
        pygame.draw.line(screen, colour, [tmp.state[0], tmp.state[1]],
                                         [x.state[0], x.state[1]], 3)
        pygame.display.update()
        # pygame.image.save(screen, "doubleint_sim{:0>3}.bmp".format(iternum))
        iternum += 1
    return x
# [0,240,230]
# #------------------------------------------------------------------------# #

# ####-------------------- Dynamics --------------------#### #

fcn = sst.fcn

# Write in the form    xdot = Ax + Bu  (linearize, if necessary)
A = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
B = np.array([[0,0],[0,0],[1,0],[0,1]])

# Define cost matrices Q,R for lqr
# Q = np.eye(sst.dim)*4
# Q[0,0] = 24
# Q[1,1] = 8
# R = np.eye(sst.udim)*29
# R[1,0] = 4
# R[0,1] = 4

Q = np.eye(sst.dim)*8
# Q[0,0] = 1
# Q[1,1] = 1
R = np.eye(sst.udim)*30

# old lqr
K, P, eigs = lqr(A, B, Q, R)


##--------------------------------##

pygame.init()
screen = pygame.display.set_mode(sst.WINSIZE)
pygame.display.set_caption('SST')
white = 255, 240, 240
screen.fill(white)
screen_regions = []
region_positions = [] # measured from top left, +x directed right, +y directed down
colour_lst = []
for i in range(num_props):
    wh = sst.regionsize_list[i]
    colour = np.array([((i) % num_props)*80,
                       ((1 + i) % num_props)*80,
                       ((2 + i) % num_props)*80])
    colour_lst.append(colour)
    screen_regions.append(pygame.Surface(wh))
    screen_regions[i].fill(colour)
    region = sst.init_coord_list[i][:2]
    region[0] = region[0] - wh[0]/2.
    region[1] = region[1] - wh[1]/2.
    region_positions.append(region)
    # "blit"/show on screen at position given by initial state of each graph in MGinit
    screen.blit(screen_regions[i], region_positions[i])
sst.obsDraw(screen, sst.OBS)
pygame.display.update()


# draw candidate trajectories, and create edgelst for each traj
index_lst = sst.goal_index_lst(num_props)
idx = 0
edgelsts = []
costs = []
for p in index_lst:
    i = p[0]
    j = p[1]
    nn = endnodes[idx]
    x0 = MG.graphs[i].V[0][0]
    colour = colour_lst[int(np.ceil(idx/2))]*0.8
    if idx == 1 or idx == 5 or idx == 2:
        while not(np.array_equal(nn.state, x0.state)):
            pygame.draw.line(screen, colour*0.8, [nn.state[0], nn.state[1]],
                             [nn.parent.state[0], nn.parent.state[1]], 6)
            nn = nn.parent
    newlst, timecost = construct_edgelist(endnodes[idx], MG.graphs[i], x0)
    edgelsts.append(newlst)
    costs.append(timecost)
    idx += 1
pygame.display.update()

# ------ Animation: regions in sequence ---------#

# r = str(np.random.randint(1000))
# os.chdir("./results/doubleint")
# os.mkdir("simulation"+r)
# os.chdir("simulation"+r)

# iternum = 0
# idx = 5
# for e in edgelsts[idx]:
#     pygame.draw.line(screen, colour_lst[int(np.ceil(idx/2))]*0.8,
#                      [e.source.state[0], e.source.state[1]],
#                      [e.target.state[0], e.target.state[1]], 6)
#     pygame.display.update()
#     pygame.image.save(screen, "doubleint_sim{:0>3}.bmp".format(iternum))
#     iternum += 1

# idx = 2
# for e in edgelsts[idx]:
#     pygame.draw.line(screen, colour_lst[int(np.ceil(idx/2))]*0.8,
#                      [e.source.state[0], e.source.state[1]],
#                      [e.target.state[0], e.target.state[1]], 6)
#     pygame.display.update()
#     pygame.image.save(screen, "doubleint_sim{:0>3}.bmp".format(iternum))
#     iternum += 1



# # ##-----------------------------------------------------------------------## #
# # Run simulation

x = MG.graphs[0].V[0][0]

# r = str(np.random.randint(1000))
# os.chdir("./results/doubleint")
# os.mkdir("simulationLQR"+r)
# os.chdir("simulationLQR"+r)

iternum = 0
count = 2
while count > 0:
    x = simulate(x, edgelsts[1] + edgelsts[5] + edgelsts[2])
    # x = simulate(x, edgelsts[0] + edgelsts[3] + edgelsts[4])
    count -= 1

# print "ccw timecost: {}".format(costs[1] + costs[5] + costs[2])
# print "cw timecost: {}".format(costs[0] + costs[3] + costs[4])

running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False
