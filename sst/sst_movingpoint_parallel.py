#!/usr/bin/env python

# point particle that is affected only by control input:
#   dictates the x- and y-acceleration (double integrator)

# Program loops until exited or until you hit SPACE.

import sys, random, math, pygame, os, time
import numpy as np
from scipy import integrate
from pygame.locals import *
from math import sqrt,cos,sin,atan2,exp
from timeit import default_timer as timer
from line_intersect import *


N = 100      # number of iterations before updating screen
num_props = 2  # number of regions (corresp. to propositions) from which to grow trees
dim = 4

#####---------- CLASSES ----------#####

class Node:
    state = np.zeros(dim)   
    cost = 0  
    parent = None
    children = []
    rep = None
    def __init__(self,statex):
         self.state = statex

class Edge:
    traj = None
    source = None
    target = None
    control = 0
    time = 0
    cost = 0
    def __init__(self, state_traj, srcnode, tarnode, controlval, t, trajcost):
        self.traj = state_traj
        self.source = srcnode
        self.target = tarnode
        self.control = controlval
        self.time = t
        self.cost = trajcost

class Graph:
    V = []
    E = []
    S = []
    def __init__(self, v, e, s):
        self.V = v
        self.E = e
        self.S = s

class MultiGraph:
    graphs = []
    def __init__(self, numgraphs, lst_init_coords):
        self.MG = []
        for i in range(numgraphs):
            xinit = Node(lst_init_coords[i])
            Vac = [xinit]
            Vinac = []
            V = [Vac, Vinac]
            E = []
            S = []
            self.graphs.append(Graph(V,E,S))


#####-------- Dynamics --------#####

# x is 4d-vector, with x = [ pos.x, pos.y, velo.x, velo.y  ]
def fcn(x,t,u):
    return np.array([x[2], x[3], u[0], u[1]])

# control space
ux = 5 # set max absolute value of control input in x-direction
uy = 5 # ... y-direction

Ux = [-ux, ux]   # min - max range of x-control  
Uy = [-uy, uy]    # min - max range of y-control  (note: pixels measured from top left, so y negated)
U = [Ux, Uy]  
Umax = reduce(lambda acc,x: max(acc,max(x)), U, U[0][0]) # used to normalize control cost (reduce is like foldl)

####---------- Constants ----------####

# planning parameters
delta_bn = 14.0 # radius in which to look for nodes which are in the tree
delta_s = 40.0  # radius of each witness node (larger => sparser)
Tprop = 1.8   # max time to propagate with one control input
epsilon = 0.99  # factor by which we reduce delta_bn/s with each loop (for SST*)
h=0.02  # numerical integration timestep

XDIM = 1200
YDIM = 600

# list state coords of all initial states (proposition regions)
init_coord_list =  [[XDIM/10., YDIM/10.,0,0],\
                    [XDIM*4/5., YDIM/8.,0,0],\
                    [XDIM*8/13., YDIM*7/8.,0,0]]

# list [width,height] of each region in same order as init_coord_list
regionsize_list = [[30.,30.],\
                   [40.,50.],\
                   [60.,20.]]

WINSIZE = [XDIM, YDIM]

# info stored in the following MultiGraph
MGinit = MultiGraph(num_props, init_coord_list)

####---------- Obstacles -----------####

# Obstacle definition
OBS=[(XDIM/3., 0, 40, YDIM/2.),\
     (XDIM/3., YDIM*4/5., 40, YDIM*1/5.),\
     (XDIM/2., YDIM*2/3., 40, YDIM*1/3.),\
     (XDIM/2., 0, 40, YDIM*2/5.),\
     (XDIM*5/7., 0, 40, YDIM*4/15.),\
     (XDIM*5/7., YDIM*11/15., 40, YDIM*7/15.)]
     
def obsDraw(screen,OBS):
    blue=(0,0,255)
    for o in OBS: 
      pygame.draw.rect(screen,blue,o)


####---------- Cost Function ----------####

# general cost fcn (may just be distance) between nodes n1, n2
def cost(n1,n2):
    dim = len(n1.state)
    total = 0
    cur = 0
    for i in range(dim):
        normfactor = 1.
        if i==1 or i==2: # normalize position
            normfactor = 30. # keep decimal to ensure floating point arithmetic used (not integer)
        else:
            normfactor = 10.
        cur = (n1.state[i]-n2.state[i])  / normfactor
        cur = cur*cur
        total += cur
    return total

##---------------------------------------##

# function to ensure trajectories stay within the defined screen (x,y state space)
def within_boundary(x):
    xval = x.state[0]
    yval = x.state[1]
    if (xval < 0 or xval > XDIM) or (yval < 0 or yval > YDIM):
        return False
    else:
        return True

# distance between Nodes n1, n2
def dist(n1, n2):
    x1 = n1.state[0]
    y1 = n1.state[1]
    x2 = n2.state[0]
    y2 = n2.state[1]
    return (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)

# distance between states p1, p2
def distxy(p1,p2):
    return (p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])

# distance (including 2D velocity space) given states p1, p2
def distxyv(p1,p2):
    return (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) \
         + (p1[2]-p2[2])*(p1[2]-p2[2]) + (p1[3]-p2[3])*(p1[3]-p2[3])

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

# returns the nearest node to newnode from the set of nodes S (based on cost, not dist)
def nearest_cost(S, newnode):
    nn = S[0]
    for p in S:
        if cost(p,newnode) < cost(nn,newnode):
            nn = p
    return nn

# outputs time duration t, as well as the new state xnew, the random control u, 
#  and the new cost 
def MonteCarlo_Prop(x, U, Tprop):
    T = h+random.random()*(Tprop-h)    # T is no smaller than h
    t = np.linspace(0,T,num = int(T/h)+1)
    u = np.zeros(len(U))
    for i in range(len(U)):  # fills each component of control vector with random value in approp. range
        u[i] = random.uniform(U[i][0],U[i][1])
    xnew = integrate.odeint(fcn, x.state, t, (u,) ) # must put args in tuple (u,) -- comma since u is the only arg
    runcost = 0
    # for i in xrange(len(xnew)-1):           
    #     runcost += distxyv(xnew[i],xnew[i+1])
    return [T,xnew,u,runcost]

# returns the node with the least root-to-node cost within delta_bn radius of a randomly chosen node,
#   or the nearest node to the random one if the delta_bn nbhd contains no nodes from S
def best_first_selection(V, delta_bn):
    r = random.random()
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


# adds a witness node if necessary (snew=newnode if it is not in delta_s radius of an existing witness),
#  and determines whether newnode is locally best given the set of witnesses, S.
#  For efficiency, returns False if not locally best, but returns the witness otherwise.
def is_locally_best(newnode, S, delta_s):
    snear = nearest(S, newnode)  # the nearest node to newnode among the vertices in S
    snew = snear
    if dist(newnode,snew) > delta_s:
        snew = newnode  # newnode is itself the best in the cost radius (snew too far to count)
        snew.rep = None;  # no rep yet as we just made a new witness
        S.append(snew)
    xpeer = snew.rep
    if xpeer == None or newnode.cost < xpeer.cost: # we've just added a new witness or newnode.cost is better than that of rep
        return snear  # returns the nearest witness node to newnode to save calculation
    else:
        return False

def is_leaf(x):
    if x == None: 
        return False
    return len(x.children) == 0
    #return x==None or len(x.children) == 0   (I think this is equivalent in Prune since None is not in Vinac)

def remove_node(V,E,badnode):
    for e in E:
        if e.source == badnode or e.target == badnode:
            if e.target == badnode: # if target, must remove node from the children of its parent
                parentnode = badnode.parent
                parentnode.children.remove(badnode)
            E.remove(e)
    V.remove(badnode)

# newnode is assumed to be locally best, snew is actually the nearest witness to newnode (found in is_locally_best)
def prune_nodes(newnode, snew, Vac, Vinac, E):
    xpeer = snew.rep  # xpeer is the representative of the nearest witness node
    if xpeer != None: # the rep is only None if it was newly added. Otherwise, it is dominated by newnode.
        Vac.remove(xpeer)
        Vinac.append(xpeer)
    snew.rep = newnode  # since newnode is assumed locally best, it becomes the rep (replaces None if new witness)
    while is_leaf(xpeer) and xpeer in Vinac:
        xparent = xpeer.parent
        remove_node(Vinac,E,xpeer)
        xpeer = xparent

def drawSolutionPath(x0,goal,V,searchradius,colour,pygame,screen):
    ## Draws solution path for each proposition region
    X = near(V,goal,searchradius)
    if X == []:
        nn = nearest_cost(V, goal)
    else:
        nn = X[0]
        for p in X:
            if cost(p, goal) < cost(nn, goal):  
                nn = p
    endnode = nn
    while nn!=x0:
        pygame.draw.line(screen, colour, [nn.state[0],nn.state[1]], [nn.parent.state[0],nn.parent.state[1]], 4)  
        nn=nn.parent
    print endnode.state
    return endnode

# def wait(iternum):
#     while True:
#         for event in pygame.event.get():
#             if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
#                 pygame.quit()
#                 sys.exit()
#             if event.type == KEYUP and event.key == K_RETURN:
#                 return [True, iternum+1]



####################################################################################################
    


def main():
    ## The following block is used to save images as the program runs. Please see block at the end as well.

    # t = time.localtime()
    # timestamp = "{:0>2}-{:0>2}-{:0>2}--{:0>2}.{:0>2}.{:0>2}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec) 
    # os.chdir("results")
    # os.chdir("sst-movingpoint")
    # os.mkdir(timestamp)
    # os.chdir(timestamp)
    # f = open("params.txt","w")
    # paramlst = ["Tprop: {}\n".format(Tprop), "delta_bn: {}\n".format(delta_bn), \
    # "delta_s: {}\n".format(delta_s), "goalbias: {}\n".format(goalbias), \
    # "screen size: {}x{}\n".format(XDIM,YDIM)]
    # f.writelines(paramlst)
    # i=1
    # for udim in U:
    #     f.writelines(["control space for dim {}: {}\n".format(i, udim)])
    #     i += 1
    # f.close()
    #-------------------------------------------------------------------------

    # setting up pygame display
    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('SST')
    white = 255, 240, 240
    black = 20, 20, 40
    green = 50,150,50
    screen.fill(white)
    screen_regions = []
    region_positions = [] # measured from top left, +x directed right, +y directed down
    colour_lst = []
    for i in range(num_props):
        wh = regionsize_list[i]
        colour = [random.random()*255,random.random()*255,random.random()*255]
        colour_lst.append(colour)
        screen_regions.append(pygame.Surface(wh))
        screen_regions[i].fill(colour)
        region = MGinit.graphs[i].V[0][0].state[:2]
        region[0] = region[0] - wh[0]/2.
        region[1] = region[1] - wh[1]/2.
        region_positions.append(region)
        # "blit"/show on screen at position given by initial state of each graph in MGinit
        screen.blit(screen_regions[i], region_positions[i]) 
    obsDraw(screen,OBS)

    MG = MGinit
    x0lst = []
    goallst = []
    for i in range(num_props):
        S = MG.graphs[i].S
        xinit = MG.graphs[i].V[0][0]
        #xinit.rep = xinit

        wh = regionsize_list[i] # [width,height] of the ith region
        goal = Node(xinit.state)  # copies xinit 
        # goal.state[0] = goal.state[0] + wh[0]/2.
        # goal.state[1] = goal.state[1] + wh[1]/2.
        goallst.append(goal)
        x0lst.append(xinit)
        S.append(xinit)

    global delta_bn, delta_s
    repeat = True
    iternum = 0
    while repeat:
        start = timer()
        for i in xrange(N):
            for j in xrange(num_props):
                Vac = MG.graphs[j].V[0]
                Vinac = MG.graphs[j].V[1]
                E = MG.graphs[j].E
                S = MG.graphs[j].S
                # print "{}".format(j) + "-"*40
                # print "Vac: {}".format(map(lambda (x): x.state, Vac))
                # print "Vinac: {}".format(map(lambda (x): x.state, Vinac))
                # print "E: {}".format(map(lambda (x): [x.source.state, x.target.state], E))

                xselected = best_first_selection(Vac, delta_bn) 
                t, xnew_traj, u, distcost = MonteCarlo_Prop(xselected, U, Tprop)
                xnew = Node(xnew_traj[-1])  # make a node from final state after propagating
                controlcost = 0
                timecost = t/Tprop # write this way as normalization, so controlcost and timecost are fractions of Umax.
                for i in range(len(u)):
                    controlcost += u[i]*u[i] / (Umax*Umax)

                # COST!!! Very important to choose appropriately TODO
                xnew.cost = xselected.cost + timecost + controlcost #+ xnew.state[2]**2 + xnew.state[3]**2

                # COLLISION CHECK
                if (check_intersect(xselected, xnew, OBS)) and within_boundary(xnew):
                    snear = is_locally_best(xnew, S, delta_s)
                    if snear:
                        xnew.parent = xselected
                        xselected.children = xselected.children + [xnew] # TODO: bad runtime! but perhaps the only way..
                        Vac.append(xnew)
                        E.append(Edge(xnew_traj, xselected, xnew, u, t, distcost))
                        prune_nodes(xnew, snear, Vac, Vinac, E)

                for e in pygame.event.get():
                    if e.type == KEYUP and e.key == K_SPACE:
                        repeat = False
                    elif e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                        sys.exit("Leaving because you requested it.")

        # blank screen and update (cannot erase lines so this is only way)
        screen.fill(white)
        for i in range(num_props):
            screen.blit(screen_regions[i], region_positions[i]) # display regions again
            E = MG.graphs[i].E
            for e in E:
                pygame.draw.line(screen,colour_lst[i],[e.source.state[0] ,e.source.state[1] ],\
                                                      [e.target.state[0] ,e.target.state[1] ])
        obsDraw(screen,OBS)
        
        endnodes = []
        for i in xrange(num_props):
            Vac = MG.graphs[i].V[0]
            Vinac = MG.graphs[i].V[1]
            V = Vac + Vinac
            colour = colour_lst[i]
            for j in xrange(num_props):
                if i == j:
                    continue
                else:
                    endnodes.append(drawSolutionPath(x0lst[i], goallst[j], V, 25., colour ,pygame, screen))

        pygame.display.update()

        ##==== SST* - decrease delta_bn and delta_s ====##
        delta_s = epsilon*delta_s
        delta_bn = epsilon*delta_bn 

        end = timer()
        print(end - start)
        
        # For saving images

        # iternum += 1
        # if iternum == 1 or iternum % 3 == 0:
        #     pygame.image.save(screen, "sstpoint{:0>2}.bmp".format(iternum))
        #-----------------------------------------------

    return MG, endnodes

if __name__ == '__main__':
    main()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False



