#!/usr/bin/env python

# Set multiple initial states (one for each ) from which to build trees

import sst_movingpoint_3 as sst
from lqr import *
import numpy as np
import pygame
from scipy import integrate
from pygame.locals import *

h = sst.h  # numercial integration timestep

# Obtain list of nodes and edges of the candidate trajectory given endnode 
def construct_edgelist(endnode, G, xinit):
    edgelst = []
    curnode =  endnode
    while curnode != xinit:
        for e in G.E:
            if e.target == curnode:
                edgelst.append(e)
                curnode = e.source
                break
    edgelst.reverse()
    return edgelst

# #------------------------------------------------------------------------# #

# Run planner to obtain candidate trajectory
G, endnode = sst.main()
print "Planning step 1 complete"
edgelst = construct_edgelist(endnode, G, sst.xinit)  # return edges and nodes of candidate traj

newxinit = endnode
# newxinit.state[2:] = [0,0] # set (new) initial velocity to 0, since LQR should bring first traj to 0 velo
newxinit.cost = 0
newgoal = sst.Node([sst.XSTART,sst.YSTART,0,0])

# the following are constants in sst_movingpoint
oldxinit = sst.xinit
sst.xinit = newxinit
sst.goalinit = newgoal
oldXGOAL = sst.XGOAL
oldYGOAL = sst.YGOAL
sst.XGOAL = newgoal.state[0]
sst.YGOAL = newgoal.state[1]

# initializing new graph Ginit which starts at xinit (i.e., endnode from prev run)
sst.sinit = sst.xinit
Vac = [sst.xinit]
Vinac = []
V = [Vac, Vinac]
E = []
S = []
sst.Ginit = sst.Graph(V,E,S)

G2, endnode2 = sst.main()
edgelst2 = construct_edgelist(endnode2, G2, sst.xinit)

#####-------- Dynamics --------#####:

# x is 4d-vector, with x = [ pos.x, pos.y, velo.x, velo.y  ]
def fcn(x,t,u):
    return np.array([x[2], x[3], u[0], u[1]])

# Write in the form    xdot = Ax + Bu  (linearize, if necessary)
A = np.matrix([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
B = np.matrix([[0,0],[0,0],[1,0],[0,1]])

# Define cost matrices Q,R for lqr
Q = np.eye(len(sst.xinit.state))*6
R = np.eye(len(sst.U))*8

K, P, eigs = solvelqr(A,B,Q,R)

# # Create list of states we want to track
# fulltraj1 = []
# fulltraj2 = []
# for e in edgelst:
#   fulltraj1.extend(e.traj)

# for e in edgelst2:
#   fulltraj2.extend(e.traj)

# TODO: How to use this to find err = x - xref (where xref is a reference state from fulltraj)
##--------------------------------##

print "Initializing simulation"
screen = pygame.display.set_mode(sst.WINSIZE)
pygame.display.set_caption('Simulated trajectory')
white = 255, 240, 240
black = 20, 20, 40
screen.fill(white)
goalregion = pygame.Surface((sst.WIDTH,sst.HEIGHT))
goalregion2 = pygame.Surface((sst.WIDTH,sst.HEIGHT))
goalregion.fill((50.0,150.0,50.0))    # fill with green colour (this is NOT the size)
goalregion2.fill((50.0,150.0,50.0)) 
screen.blit(goalregion, (oldXGOAL,oldYGOAL))
screen.blit(goalregion2, (sst.XGOAL,sst.YGOAL))
sst.obsDraw(screen,sst.OBS)
# draw chosen trajectories
pink = 200, 20, 240
blue = 20, 100, 200
nn = endnode
while nn!=oldxinit:
    pygame.draw.line(screen,pink, [nn.state[0],nn.state[1]], [nn.parent.state[0],nn.parent.state[1]] ,5)  
    nn=nn.parent
nn = endnode2
while nn!=newxinit:
    pygame.draw.line(screen,blue, [nn.state[0],nn.state[1]], [nn.parent.state[0],nn.parent.state[1]] ,5)  
    nn=nn.parent

###-------------------------------------------------------------------------###
x = edgelst[0].source 
i=5
while i>0:
    # count = 0
    # threshold = int(3./4.*len(edgelst))
    # b = len(edgelst) - threshold # how many steps from threshold til the end of the traj
    # trigger = True
    for e in edgelst+edgelst2:
        # if count > len(edgelst) and trigger:
        #     threshold = int(3./4.*len(edgelst2))
        #     b = len(edgelst2) - threshold
        #     count = 0
        #     trigger = False

        # if count < threshold: 
        #     err = np.array(x.state) - np.array(e.target.state)
        # else: # when past threshold, start scaling velocity down quadratically
        #     xx = count - threshold  # number of steps since threshold was passed
        #     mult_factor = -1/(b*b)*(xx - b)*(xx + b)  # when xx reaches the end, this is 0. Initially 1.
        #     M = np.eye(len(x.state))
        #     M[2:] *= mult_factor
        #     target_adjusted = np.dot(M, np.array(e.target.state))
        #     err = np.array(x.state) - target_adjusted
        T = e.time
        t = np.linspace(0,T,num = int(T/h)+1)
        tmp = x
        for now in range(len(t)-1):
            tt = np.linspace(t[now], t[now]+h, num = 2) # integrate forward one step
            err = np.array(x.state) - np.array(e.traj[now+1])
            u = np.array(-np.dot(K,err))[0]
            for ind in range(len(u)):
                if abs(u[ind]) > sst.Umax:
                    u[ind] = u[ind]/abs(u[ind]) * sst.Umax
            # print "u: {}".format(u)
            x = integrate.odeint(fcn, x.state, t, (u,) )
            x = sst.Node(x[-1])

        #err = np.array(x.state) - np.array(e.target.state)
        #u = np.array(-np.dot(K,err))[0] # weird python thing where the matrix mult returns a matrix, not array    

        # for ind in range(len(u)):
        #     if abs(u[ind]) > sst.Umax:
        #         u[ind] = u[ind]/abs(u[ind]) * sst.Umax
        # print(" u: {}\n err: {}".format(u,err))
        # t = np.linspace(0,T,num = int(T/h)+1)
        # tmp = x
        # x = integrate.odeint(fcn, x.state, t, (u,) )
        # x = sst.Node(x[-1])
        pygame.draw.line(screen,black,[tmp.state[0], tmp.state[1]],
                                      [x.state[0] , x.state[1]])
        pygame.display.update()

        # count +=1
    i-=1

running = True
while running:
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False


