#!/usr/bin/env python

# rrtstar.py
# This program generates a 
# asymptotically optimal rapidly exploring random tree (RRT* proposed by Sertac Keraman, MIT) in a rectangular region.
#
# Originally written by Steve LaValle, UIUC for simple RRT in
# May 2011
# Modified by Md Mahbubur Rahman, FIU for RRT* in
# January 2016

import sys, random, math, pygame
from pygame.locals import *
from math import sqrt,cos,sin,atan2,exp

#constants
NUMNODES = 1000
XDIM = 640
YDIM = 480
XGOAL = 560.0
YGOAL = 420.0
# randomize area of goal region but make sure it's large enough using sigmoid fcn
WIDTH = (1/(1+exp(-2*random.random())) - 0.2)*(XDIM-XGOAL)
HEIGHT = (1/(1+exp(-2*random.random())) - 0.2)*(YDIM-YGOAL)
WINSIZE = [XDIM, YDIM]
EPSILON = 15.0

RADIUS=10.0

def dist(p1,p2):
    return (p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])

def step_from_to(p1,p2):
    stepsize = random.random()*EPSILON
    if dist(p1,p2) < stepsize:
        return p2
    else:
        theta = atan2(p2[1]-p1[1],p2[0]-p1[0])
        return p1[0] + stepsize*cos(theta), p1[1] + stepsize*sin(theta)

def chooseParent(nn,newnode,nodes):
    for p in nodes:
        pdist = dist([p.x,p.y],[newnode.x,newnode.y])
        nndist = dist([nn.x,nn.y],[newnode.x,newnode.y])
        if  pdist<RADIUS and p.cost+pdist < nn.cost+nndist:
            nn = p
    newnode.cost=nn.cost+dist([nn.x,nn.y],[newnode.x,newnode.y])
    newnode.parent=nn
    return newnode,nn

def reWire(nodes,newnode,pygame,screen):
    white = 255, 240, 200
    black = 20, 20, 40
    for i in xrange(len(nodes)):
        p = nodes[i]
        pdist = dist([p.x,p.y],[newnode.x,newnode.y])
        if p!=newnode.parent and pdist <RADIUS and newnode.cost+pdist < p.cost:
            pygame.draw.line(screen,white,[p.x,p.y],[p.parent.x,p.parent.y])  
            p.parent = newnode
            p.cost=newnode.cost+dist([p.x,p.y],[newnode.x,newnode.y]) 
            nodes[i]=p  
            pygame.draw.line(screen,black,[p.x,p.y],[newnode.x,newnode.y])                    
    return nodes

def drawSolutionPath(start,goal,nodes,pygame,screen):
    ## Draws solution path to a node that is found within the goal region
    pink = 200, 20, 240
    nn = nodes[0]
    for p in nodes:
        # find closest node to the goal (centre of goal region)
        if dist([p.x,p.y],[goal.x,goal.y]) < dist([nn.x,nn.y],[goal.x,goal.y]):
            nn = p
    while nn!=start:
        pygame.draw.line(screen,pink,[nn.x,nn.y],[nn.parent.x,nn.parent.y],5)  
        nn=nn.parent



class Node:
    x = 0
    y = 0
    cost=0  
    parent=None
    def __init__(self,xcoord, ycoord):
         self.x = xcoord
         self.y = ycoord
    
def main():
    #initialize and prepare screen
    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('RRTstar')
    white = 255, 240, 200
    black = 20, 20, 40
    screen.fill(white)
    goalregion = pygame.Surface((WIDTH,HEIGHT))
    goalregion.fill((50.0,150.0,50.0))
    screen.blit(goalregion, (XGOAL,YGOAL))

    nodes = []
    
    #nodes.append(Node(XDIM/2.0,YDIM/2.0)) # Start in the center
    nodes.append(Node(10.0,10.0)) # Start in the corner
    start=nodes[0]
    goal=Node(XGOAL,YGOAL)
    for i in range(NUMNODES):
        rand = Node(random.random()*XDIM, random.random()*YDIM)
        nn = nodes[0]
        numselect = int(len(nodes))
        random.shuffle(nodes)
        for p in nodes[:numselect]:
            if dist([p.x,p.y],[rand.x,rand.y]) < dist([nn.x,nn.y],[rand.x,rand.y]):
                nn = p
        interpolatedNode= step_from_to([nn.x,nn.y],[rand.x,rand.y])
        newnode = Node(interpolatedNode[0],interpolatedNode[1])
        [newnode,nn]=chooseParent(nn,newnode,nodes);  
        nodes.append(newnode)
        pygame.draw.line(screen,black,[nn.x,nn.y],[newnode.x,newnode.y])
        nodes=reWire(nodes,newnode,pygame,screen)
        # if (XGOAL <= newnode.x and newnode.x <= XGOAL + WIDTH) and \
        # (YGOAL <= newnode.y and newnode.y <= YGOAL + HEIGHT):
        #     goal = newnode
        #     break

        if i%200 == 0:
            pygame.display.update()

        #print i, "    ", nodes

        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                sys.exit("Leaving because you requested it.")
    goal.x = XGOAL+WIDTH/2
    goal.y = YGOAL+HEIGHT/2
    drawSolutionPath(start,goal,nodes,pygame,screen)
    pygame.display.update()
# if python says run, then we should run
if __name__ == '__main__':
    main()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False



