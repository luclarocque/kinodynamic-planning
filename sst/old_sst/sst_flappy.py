#!/usr/bin/env python

# Program loops until exited or until you hit SPACE.

import sys, random, math, pygame, os, time
import numpy as np
from scipy import integrate
from pygame.locals import *
from math import sqrt,cos,sin,atan2,exp
from timeit import default_timer as timer
from line_intersect import *

#constants
XDIM = 1600
YDIM = 900
XSTART = 160.0
YSTART = 120.0
XGOAL = XDIM*4/5 + random.random()*XDIM/5
YGOAL = random.random()*YDIM/3
WIDTH = (0.1*random.random()+0.3)*(XDIM-XGOAL) # between 0.3 to 0.4 times XDIM-XGOAL
HEIGHT = (1/(1+exp(-0.2*random.random())) - 0.2)*(YDIM-YGOAL)
WINSIZE = [XDIM, YDIM]

N = 500
delta_bn = 12.0
delta_s = 5.0
Tprop = 2.2
goalbias = 0.05
xinit = np.array([XSTART, YSTART, 0.0, 0.0]) # initialize position and velocity

# control space
Ux = [-5.0, 5.0]   # min - max range of x-control  
Uy = [-30.0, 0.0]    # min - max range of y-control  (note: pixels measured from top left, so y negated)
U = [Ux, Uy]  

# Obstacle definition
OBS=[(XDIM/3, 0, 40, YDIM/2),\
     (XDIM/3, YDIM*4/5, 40, YDIM*1/5),\
     (XDIM/2, YDIM*2/3, 40, YDIM*1/3),\
     (XDIM/2, 0, 40, YDIM*2/5),\
     (XDIM*5/7, 0, 40, YDIM*3/7),\
     (XDIM*5/7, YDIM*4/7, 40, YDIM*3/7)]
     
def obsDraw(screen,OBS):
    blue=(0,0,255)
    for o in OBS: 
      pygame.draw.rect(screen,blue,o)


#GLOBAL VARIABLES

t = [] # array of time linspace


##### Dynamics #####:
#Constants:
m=1
g=9.8

# x is 4d-vector, with x = [ pos.x, pos.y, velo.x, velo.y  ]
def fcn(x,t,u):
    return np.array([x[2], x[3], u[0], g+u[1]])

def within_boundary(x):
    xval = x.state[0]
    yval = x.state[1]
    if (xval < 0 or xval > XDIM) or (yval < 0 or yval > YDIM):
        return False
    else:
        return True

# distance given Nodes n1, n2
def dist(n1, n2):
    x1 = n1.state[0]
    y1 = n1.state[1]
    x2 = n2.state[0]
    y2 = n2.state[1]
    return (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)

# distance given states p1, p2
def distxy(p1,p2):
    return (p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])

# returns list of nodes within delta_bn radius of newnode from the set of nodes S
def near(S, newnode, delta_bn):
    X = []
    for p in S:
        if dist(p, newnode) < delta_bn:
            X.append(p)
    return X

# returns the nearest node to newnode from the set of nodes S
def nearest(S, newnode):
    nn = S[0]
    for p in S:
        if dist(p,newnode) < dist(nn,newnode):
            nn = p
    return nn

# outputs time duration t, as well as the new state xnew, the random control u, 
#  and the new cost 
def MonteCarlo_Prop(x, U, Tprop, h=0.1):
    T = h+random.random()*(Tprop-h)    # T is no smaller than h
    t = np.linspace(0,T,num = int(T/h))
    u = np.zeros(len(U))
    for i in range(len(U)):  # fills each component of control vector with random value in approp. range
        u[i] = random.uniform(U[i][0],U[i][1])
    ####### HEURISTIC ########    
    if random.random() < 0.65:
        if x.state[2] > 0: u[0] = abs(u[0]) # heuristic to "continue" x-momentum
        else: u[0] = -abs(u[0])
    ##########################
    xnew = integrate.odeint(fcn, x.state, t, (u,) ) # must put args in tuple (u,) -- comma since u is the only arg
    runcost = 0
    for i in xrange(len(xnew)-1):
        runcost += distxy(xnew[i],xnew[i+1])
    return [t,xnew,u,runcost]

# returns the node with the least root-to-node cost within delta_bn radius of a randomly chosen node,
#   or the nearest node to the random one if the delta_bn nbhd contains no nodes from S
def best_first_selection(V, delta_bn, goalbias):
    r = random.random()
    if r < goalbias:  # sample in goal region a certain % of the time
        rand = Node([XGOAL + random.random()*WIDTH, YGOAL + random.random()*HEIGHT,0,0])
    ####### HEURISTIC ########
    elif len(V) < 36:
        rand = Node([XSTART + 10*cos(len(V)), YSTART + 10*sin(len(V)), 0 , 0])
    ##########################
    else:
        rand = Node([random.random()*XDIM, random.random()*YDIM,0,0])
    Xnear = near(V, rand, delta_bn)
    if Xnear == []:
        return nearest(V, rand)  # return nearest existing node in V since no nodes in Xnear.
    else:
        nn = Xnear[0]
        for p in Xnear:
            if p.cost < nn.cost:
                nn = p
        return nn  # return node in Xnear with least cost from root to node


# adds a witness node if necessary (newnode not in delta_s radius of an existing witness),
#  and determines whether newnode is locally best given the set of witnesses, S.
#  For efficiency, returns False if not locally best, but returns the new witness otherwise.
def is_locally_best(newnode, S, delta_s):
    snew = nearest(S, newnode)
    if dist(newnode,snew) > delta_s:
        snew = newnode
        snew.rep = None; 
        S.append(snew)
    xpeer = snew.rep
    if xpeer == None or newnode.cost < xpeer.cost:
        return snew
    else:
        return False

def is_leaf(x):
    # return len(x.children) == 0
    return x==None or len(x.children) == 0

def remove_node(V,E,badnode):
    for e in E:
        if e.source == badnode or e.target == badnode:
            if e.target == badnode:
                parentnode = badnode.parent
                parentnode.children.remove(badnode)
            E.remove(e)
    V.remove(badnode)

# newnode is assumed to be locally best
def prune_nodes(newnode, snew, Vac, Vinac, E):
    xpeer = snew.rep  # xpeer is the representative of the nearest witness node
    if xpeer != None:
        Vac.remove(xpeer)
        Vinac.append(xpeer)
    snew.rep = newnode
    while is_leaf(xpeer) and xpeer in Vinac:
        xparent = xpeer.parent
        remove_node(Vinac,E,xpeer)
        xpeer = xparent

def drawSolutionPath(x0,goal,V,pygame,screen):
    ## Draws solution path to a node that is found within the goal region
    pink = 200, 20, 240
    X = near(V,goal,max(WIDTH,HEIGHT))
    if X == []:
        nn = nearest(V, goal)
    else:
        nn = X[0]
        for p in X:
            if p.cost < nn.cost and (XGOAL <= p.state[0] and p.state[0]<= XGOAL + WIDTH) \
                                and (YGOAL <= p.state[1] and p.state[1]<= YGOAL + HEIGHT):
                nn = p
    while nn!=x0:
        pygame.draw.line(screen,pink, [nn.state[0],nn.state[1]], [nn.parent.state[0],nn.parent.state[1]] ,5)  
        nn=nn.parent

# def wait(iternum):
#     while True:
#         for event in pygame.event.get():
#             if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
#                 pygame.quit()
#                 sys.exit()
#             if event.type == KEYUP and event.key == K_RETURN:
#                 return [True, iternum+1]

##### CLASSES #####

class Node:
    state = [0,0,0,0]   
    cost=0  
    parent=None
    children = []
    rep=None
    def __init__(self,statex):
         self.state = statex

class Edge:
    source = None
    target = None
    control = 0
    time = 0
    cost = 0
    def __init__(self, srcnode, tarnode, controlval, t, arclength):
         self.source = srcnode
         self.target = tarnode
         self.control = controlval
         self.time = t
         self.cost = arclength

#####################
    
def main():
    t = time.localtime()
    timestamp = "{:0>2}-{:0>2}-{:0>2}--{:0>2}.{:0>2}.{:0>2}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec) 
    os.chdir("results")
    os.chdir("sst-flappy")
    os.mkdir(timestamp)
    os.chdir(timestamp)
    f = open("params.txt","w")
    paramlst = ["Tprop: {}\n".format(Tprop), "delta_bn: {}\n".format(delta_bn), \
    "delta_s: {}\n".format(delta_s), "goalbias: {}\n".format(goalbias), \
    "screen size: {}x{}\n".format(XDIM,YDIM)]
    f.writelines(paramlst)
    i=1
    for udim in U:
        f.writelines(["control space for dim {}: {}\n".format(i, udim)])
        i += 1
    f.close()

    
    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('RRTstar')
    white = 255, 240, 240
    black = 20, 20, 40
    screen.fill(white)
    goalregion = pygame.Surface((WIDTH,HEIGHT))
    goalregion.fill((50.0,150.0,50.0))    # fill with green colour (this is NOT the size)
    screen.blit(goalregion, (XGOAL,YGOAL))
    obsDraw(screen,OBS)


    x0 = Node([XSTART, YSTART,0,0]) # initially at rest
    x0.cost = 0.

    goal = Node([XGOAL,YGOAL,0,0])

    s0 = x0
    s0.rep = x0

    Vac = [x0]
    Vinac = []
    E = []
    S = [s0]

    repeat = True
    iternum = 0
    while repeat:
        start = timer()
        for i in xrange(N):
            xselected = best_first_selection(Vac, delta_bn, goalbias) 
            t, xnew_traj, u, newcost = MonteCarlo_Prop(xselected, U, Tprop)
            xnew = Node(xnew_traj[-1])  # make a node from final state after propagating
            xnew.cost = xselected.cost + newcost

            # COLLISION CHECK
            if (check_intersect(xselected, xnew, OBS)) and within_boundary(xnew):
                snew = is_locally_best(xnew, S, delta_s)
                if snew:
                    xnew.parent = xselected
                    xselected.children = xselected.children + [xnew]
                    Vac.append(xnew)
                    E.append(Edge(xselected, xnew, u, t, newcost))

                    prune_nodes(xnew, snew, Vac, Vinac, E)
                
            for e in pygame.event.get():
                if e.type == KEYUP and e.key == K_SPACE:
                    repeat = False
                elif e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                    sys.exit("Leaving because you requested it.")

        screen.fill(white)
        screen.blit(goalregion, (XGOAL,YGOAL))
        obsDraw(screen,OBS)
        for e in E:
            pygame.draw.line(screen,black,[e.source.state[0] ,e.source.state[1] ],\
                                          [e.target.state[0] ,e.target.state[1] ])
        goal.state[0] = XGOAL+WIDTH/2
        goal.state[1] = YGOAL+HEIGHT/2
        V = Vac + Vinac
        drawSolutionPath(x0,goal,V,pygame,screen)
        pygame.display.update()

        end = timer()
        print(end - start)
        
        iternum += 1
        pygame.image.save(screen, "sstflappy{:0>2}.bmp".format(iternum))



if __name__ == '__main__':
    main()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False



