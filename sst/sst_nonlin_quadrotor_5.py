#!/usr/bin/env python

# finds trajectories between every pair of props
# Program loops until exited or until you hit SPACE.

# This version (5) has the following fixes:
#   - fixes an error in fcn, the dynamics of the system: g18 term was negated.
#   - best-first selection now works on all of V=Vac+Vinac, not just Vac.

import sys, pygame, time
import numpy as np
from scipy import integrate
from pygame.locals import *
from math import sqrt, cos, sin, tan, exp
from timeit import default_timer as timer
from line_intersect import check_intersect
from save_obj import save_object
import time

# np.random.seed(125)
np.random.seed(126)

np.set_printoptions(suppress=True)
sys.setrecursionlimit(500000)
# ###---------- PARAMETERS ----------### #


N = 100      # number of iterations before updating screen
num_props = 2  # number of regions (propositions) from which to grow trees
dim = 12
udim = 4
posdim = 3  # number of spatial dimensions (e.g., 2D, 3D)

# planning parameters
delta_bn = 20.0  # radius in which to look for nodes which are in the tree
delta_s = 15.0  # radius of each witness node (larger => sparser)
searchradius = 60.
Tprop = 1.2   # max time to propagate with one control input
epsilon = 0.999  # factor by which we reduce delta_bn/s each loop (for SST*)
h = 0.05  # numerical integration time step

DIMsize = [800, 500, 200]  # [XDIM, YDIM (, ZDIM)]
# DIMsize = [1920, 1080, 200]

# ----------------------- #
XDIM = DIMsize[0]
YDIM = DIMsize[1]
ZDIM = 0
if posdim > 2:
    ZDIM = DIMsize[2]

WINSIZE = [XDIM, YDIM]

# ----------------------- #

# list state coords of all initial states (proposition regions)
init_coord_list = \
  [np.array([XDIM*1/10., YDIM*1/2., ZDIM/4., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
   np.array([XDIM*9/10., YDIM*2/9., ZDIM/2., -3/4*np.pi, 0., 0., 0., 0., 0., 0., 0., 0.]),
   np.array([XDIM*8/13., YDIM*7/8., 0.,  0., 0., 0., 0., 0., 0., 0., 0., 0.])]

# list [width,height] of each region in same order as init_coord_list
regionsize_list = [[30., 40.],
                   [40., 30.],
                   [40., 40.]]

colour_lst = (np.array([170, 20, 20]),
              np.array([20, 20, 170]),
              np.array([10, 240, 10]))

# set maximum velocity
vranges = [[-50., 50.], [-50., 50.], [-50., 50.]]

# control space (make sure this is same size as udim)
U = np.array([[-10., 0.],            [-0.000008, 0.000008],
              [-0.000008, 0.000008], [-0.00001, 0.00001]])

R = np.eye(udim)  # controlcost matrix R, which appears as u'Ru
rho = 4e9

R[0, 0] = rho*1e-13
R[1, 1] = rho
R[2, 2] = rho
R[3, 3] = rho

# used to scale error when determining cost between nodes
multfactor = np.array([[6., 6., 5., .5, .5, .5, .7, .7, .7, .7, .7, .7]]).T
# multfactor = np.array([[2.5, 2.5, 2.5, 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T
divfactor = 100.*np.linalg.norm(multfactor)**2.
k_T = 4.
k_c = 6.


# Umax = reduce(lambda acc,x: max(acc,max(x)), U, U[0][0])

# ###---------- CLASSES ----------### #


class Node:
    state = np.zeros(dim)
    cost = 0
    parent = None
    children = []
    rep = None

    def __init__(self, statex):
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
        for i in range(numgraphs):
            xinit = Node(lst_init_coords[i])
            Vac = [xinit]
            Vinac = []
            V = [Vac, Vinac]
            E = []
            S = []
            self.graphs.append(Graph(V, E, S))

# info stored in the following MultiGraph
MGinit = MultiGraph(num_props, init_coord_list)

# ###-------- Dynamics --------### #
m = 0.30
g = 9.8
Ix = 0.00001395
Iy = 0.00001436
Iz = 0.00002173

"""
state x = [[x, y, z, psi, theta, phi, xdot, ydot, zdot, p, q, r]].T
  where:
    x, y, z          -- position (World frame)
    psi, theta, phi  -- yaw, pitch, roll  (World frame)
    xdot, ydot, zdot -- translational velocity (World frame)
    p, q, r          -- angular velocities (Body frame)

    (u, v, w          -- translational velocity (Body frame))

    Note that we approximate [phidot, thetadot, psidot] == [p, q, r]


"""


def fcn(t, x, u):
    g17 = -1./m * ( sin(x[5])*sin(x[3]) + cos(x[5])*cos(x[3])*sin(x[4]) )
    g18 = -1./m * ( cos(x[5])*sin(x[3])*sin(x[4]) - sin(x[5])*cos(x[3]) )
    g19 = -1./m * ( cos(x[5])*cos(x[4]) )

    g1 = np.array([0., 0., 0., 0., 0., 0., g17, g18, g19, 0., 0., 0.])
    g2 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1./Ix, 0., 0.])
    g3 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1./Iy, 0])
    g4 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1./Iz])
    G = np.array([g1, g2, g3, g4]).T

    f1 = x[6]
    f2 = x[7]
    f3 = x[8]
    f4 = x[10]*sin(x[5])/cos(x[4]) + x[11]*cos(x[5])/cos(x[4])
    f5 = x[10]*cos(x[5]) - x[11]*sin(x[5])
    f6 = x[9] + x[10]*sin(x[5])*tan(x[4]) + x[11]*cos(x[5])*tan(x[4])
    f7 = 0.
    f8 = 0.
    f9 = -g
    f10 = x[10]*x[11]*(Iy-Iz)/Ix
    f11 = x[9]*x[11]*(Iz-Ix)/Iy
    f12 = x[9]*x[10]*(Ix-Iy)/Iz
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]) \
        + np.dot(G, u)


# returns the A matrix (Jacobian) for LQR. Use eqm=0 to NOT approximate
#   with the constant matrix at equilibrium.
def A(t, x, eqm='constant matrix'):
    if eqm == 0:
        x, y, z, ps, th, ph, u, v, w, p, q, r = x
        ps = fix_angle(ps)
        th = fix_angle(th)
        ph = fix_angle(ph)

        # redefine u, v, w
        # Given are the state values for xdot, ydot, zdot, so we obtain these
        # values in the body-fixed frame by multiplying by rotation matrix R
        U = np.dot(R(ps, th, ph), np.array([[u, v, w]]).T)
        u, v, w = U

        Acols_012 = np.zeros((12, 3))

        A = np.array(
            [[w*ph-v, w + v*ph, w*ps+v*th, 1, ph*th-ps, ph*ps+th, 0, 0, 0],
             [th*(v*ph+w)+u, ps*(v*ph+w), v*ps*th-w, ps, 1+ph*th*ps, ps*th-ph, 0, 0, 0],
             [0, -u, v, -th, ph, 1, 0, 0, 0],
             [0, 0, q, 0, 0, 0, 0, ph, 1],
             [0, 0, -r, 0, 0, 0, 0, 1, -ph],
             [0, r+q*ph, q*th, 0, 0, 0, 1, ph*th, th],
             [0, g, 0, 0, r, -q, 0, -w, v],
             [0, 0, -g, -r, 0, p, w, 0, -u],
             [0, 0, 0, q, -p, 0, -v, u, 0],
             [0, 0, 0, 0, 0, 0, 0, r*(Iy - Iz)/Ix, q*(Iy - Iz)/Ix],
             [0, 0, 0, 0, 0, 0, r*(Iz-Ix)/Iy, 0, p*(Iz-Ix)/Iy],
             [0, 0, 0, 0, 0, 0, q*(Ix-Iy)/Iz, p*(Ix-Iy)/Iz, 0]
             ])
        A = np.append(Acols_012, A, axis=1)

        # # quick fix for when things blow up...
        # if np.any(A > 2000):
        #     A = np.zeros([dim, dim])
        #     A[0:3, 6:9] = np.eye(3)
        #     A[3:6, 9:12] = np.fliplr(np.eye(3))
        #     A[6:8, 4:6] = np.array([[-g, 0], [0, g]])

        # print "df/dx = A:\n", A
    else:
        A = np.zeros([dim, dim])
        A[0:3, 6:9] = np.eye(3)
        A[3:6, 9:12] = np.fliplr(np.eye(3))
        A[6:8, 4:6] = np.array([[-g, 0], [0, g]])

    return A


# ###---------- Obstacles -----------### #

# List of Obstacles
OBS = [(XDIM/4., YDIM*1/9, 100, YDIM*2/9.),
       (XDIM/3., YDIM*3/5., 80, YDIM*2/7.),
       (XDIM*6/11., YDIM*2/9, 90, YDIM*1/5.),
       (XDIM*7/10., YDIM*7/11., 120, YDIM*4/10.)]


def obsDraw(screen, OBS):
    colour = (70, 70, 70)
    for o in OBS:
        o = pygame.Rect(o).inflate(-1, -1)
        pygame.draw.rect(screen, colour, o)
        pygame.draw.rect(screen, [0, 0, 0], o, 1)


# ###---------- Cost Function ----------### #

# general cost fcn (may just be distance) between nodes n1, n2
def cost(n1, n2):
    x1 = np.array(n1.state).reshape(dim, 1)
    x2 = np.array(n2.state).reshape(dim, 1)
    tot = (x1-x2) * multfactor  # elementwise product
    tot = np.dot(tot.T, tot)/divfactor
    return float(tot)

# #---------------------------------------# #


# maps real-valued angle a to the interval [-pi, pi]
def fix_angle(a):
    a = np.mod(a, 2*np.pi)
    if a > np.pi:
        a -= 2*np.pi
    return a


# x is a state (tuple), rescales angles using fix_angle function
def fix_state(x):
    a = [fix_angle(xi) for xi in x[3:6]]
    x[3:6] = np.array(a)
    return x


# ensures trajectories stay within the defined screen (x,y state space)
def within_boundary(x):
    for i in range(posdim):
        pos = x.state[i]
        if not(0 <= pos <= DIMsize[i]):
            return False
    return True


# distance between Nodes n1, n2
def dist(n1, n2):
    total = 0
    for i in range(posdim):
        pos1 = n1.state[i]
        pos2 = n2.state[i]
        total += (pos1-pos2)*(pos1-pos2)
    return total


# returns list of nodes within delta_bn radius of newnode from the set S
def near(S, newnode, searchradius):
    X = []
    for p in S:
        if dist(p, newnode) < searchradius*searchradius:
            X.append(p)
    return X


# returns the nearest (dist) node to newnode from the set of nodes S
def nearest(S, newnode):
    nn = S[0]
    for p in S:
        if dist(p, newnode) < dist(nn, newnode):
            nn = p
    return nn


# returns the nearest node to targetnode from the set S (based on cost)
def nearest_cost(S, targetnode):
    nn = S[0]
    for p in S:
        if cost(p, targetnode) < cost(nn, targetnode):
            nn = p
    return nn


# outputs time duration t, as well as the new state xnew, the random control u,
#  and the new cost
def MonteCarlo_Prop(x, U, Tprop):
    # T = h + np.random.rand()*(Tprop-h)    # T is no smaller than h
    T = Tprop/4. + np.random.rand()*(Tprop-h)
    u = np.zeros(udim)
    # fill each component of control vector with random value in approp. range
    for i in range(udim):
        u[i] = np.random.uniform(U[i][0], U[i][1])
    ode = integrate.ode(fcn, A)  # TODO: much faster with Jac (A). Accurate?
    ode.set_initial_value(x.state, 0).set_f_params(u)
    ode.set_integrator('vode', nsteps=2000, method='bdf')
    ts = []
    newtraj = [x.state]
    while ode.successful() and ode.t < T:
        ode.integrate(ode.t + h)
        ts.append(ode.t)
        newtraj.append(ode.y)
    runcost = 0
    return [T, newtraj, u, runcost]


# returns the node with the least root-to-node cost within delta_bn radius
#   of a randomly chosen node, or the nearest node to the random one if the
#   delta_bn nbhd contains no nodes from S
def best_first_selection(V, delta_bn):
    r = np.random.rand()
    randstate = np.random.rand(posdim) * DIMsize
    rand = Node(randstate)
    Xnear = near(V, rand, delta_bn)
    if Xnear == []:
        return nearest(V, rand)  # return nearest existing node in V
    else:
        nn = Xnear[0]
        for p in Xnear:
            if p.cost < nn.cost:
                nn = p
        return nn  # return node in Xnear with least cost from root to node


# adds a witness node if necessary (snew=newnode if it is not in delta_s radius
#  of an existing witness), and determines whether newnode is locally best
#  given the set of witnesses, S.
#  Returns False if not locally best, but returns the witness otherwise.
def is_locally_best(newnode, S, delta_s):
    snear = nearest(S, newnode)  # the nearest node to newnode among the vertices in S
    snew = snear
    if dist(newnode, snew) > delta_s*delta_s:
        snew = newnode  # newnode is itself the best in the cost radius (snear too far to count)
        snew.rep = None;  # no rep yet as we just made a new witness
        S.append(snew)
    xpeer = snew.rep
    if xpeer is None or newnode.cost < xpeer.cost: # we've just added a new witness or newnode.cost is better than that of rep
        return snear  # returns the nearest witness node to newnode to save calculation
    else:
        return False


def is_leaf(x):
    if x is None:
        return False
    return len(x.children) == 0


def remove_node(V, E, badnode):
    for e in E:
        if e.source == badnode or e.target == badnode:
            # if target, must remove node from the children of its parent
            if e.target == badnode:
                parentnode = badnode.parent
                parentnode.children.remove(badnode)
            E.remove(e)
    V.remove(badnode)


# newnode is assumed to be locally best, snew is actually the nearest witness
#   to newnode (found in is_locally_best(...))
def prune_nodes(newnode, snew, Vac, Vinac, E):
    xpeer = snew.rep  # xpeer is the representative of the nearest witness node
    if xpeer is not None:  # not newly added, thus it is dominated by newnode.
        Vac.remove(xpeer)
        Vinac.append(xpeer)
    # newnode locally best => becomes the rep (replaces None if new witness)
    snew.rep = newnode
    while is_leaf(xpeer) and xpeer in Vinac:
        xparent = xpeer.parent
        remove_node(Vinac, E, xpeer)
        xpeer = xparent


# returns multiplying factor to scale colour with cost. Bright = low cost.
def colour_mult(tarnode, colour, least_cost, max_cost):
    col = colour*(1.8 - 1.6*((tarnode.cost - least_cost)/(max_cost-least_cost+1.))**(1/3.))
    col = map(lambda x: min(x, 255), col)
    return col

# Draws solution path for each proposition region
#   x0: initial node
#   goal: goal node
#   V: set of vertices in which to search for best path
#   searchradius: used to find nodes within searchradius of the goal
#   colour: array/tuple (R,G,B)
#   l: least cost
#   m: max cost
def drawSolutionPath(x0, goal, V, searchradius, colour, l, m, pygame, screen):
    X = near(V, goal, searchradius)
    if X == []:
        X = V
        # nn = nearest_cost(V, goal)
    nn = X[0]
    for p in X:
        # if cost(p, goal) < cost(nn, goal):
        if p.cost < nn.cost:
            nn = p
    endnode = nn
    colour = colour_mult(endnode, colour, l, m)
    while nn != x0:
        pygame.draw.line(screen, colour, [nn.state[0], nn.state[1]],
                         [nn.parent.state[0], nn.parent.state[1]], 6)
        nn = nn.parent
    print("     pos: {}".format(endnode.state[0:3]))
    print("  angles: {}".format(endnode.state[3:6]))
    print("    velo: {}".format(endnode.state[6:9]))
    print("ang velo: {}".format(endnode.state[9:12]))
    print("\nCost: {}".format(endnode.cost))
    print("least_cost: {}, max_cost: {}".format(l, m))
    print("___________________")
    return endnode

# Returns a list of pairs (i,j), where i,j are indices of x0lst and goallst
#  Indicates that the first graph (ie index 0 of the output) goes from 0 to 1
def goal_index_lst(num_props):
    lst = []
    for i in xrange(num_props):
        for j in xrange(num_props):
            if i != j:
                lst.append((i, j))
    return lst


# ########################################################################## #


def main():
    # The following block is used to save images as the program runs. Please see block at the end as well.

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

    main_start = timer()

    # setting up pygame display
    pygame.init()
    screen = pygame.display.set_mode(WINSIZE)
    pygame.display.set_caption('SST')
    white = 255, 240, 240
    black = 20, 20, 40
    green = 50, 150, 50
    screen.fill(white)
    rects = []  # list of rectangles for redrawing
    for i in range(num_props):
        wh = regionsize_list[i]  # pair (width, height)
        colour = colour_lst[i]
        region = init_coord_list[i][:2]  # take x, y values from init state [i]
        # adjust so goal in centre of region by subtracting half of wh[i]
        border_rect = pygame.Rect(region[0] - wh[0]/2., region[1] - wh[1]/2.,
                                  wh[0], wh[1])
        rects.append(border_rect)
        pygame.draw.rect(screen, colour, border_rect)
        pygame.draw.rect(screen, black, border_rect, 1)

    obsDraw(screen, OBS)
    pygame.display.update()

    # setup x0lst, goallst (and initialize S for each graph in MG)
    MG = MGinit
    x0lst = []
    goallst = []
    for i in range(num_props):
        S = MG.graphs[i].S
        xinit = MG.graphs[i].V[0][0]
        goal = Node(xinit.state)  # copies xinit (not the same node)
        goallst.append(goal)
        x0lst.append(xinit)
        S.append(xinit)

    # initialize cost
    index_lst = goal_index_lst(num_props)
    for p in index_lst:
        i = p[0]
        j = p[1]
        xinit = x0lst[i]
        goal = goallst[j]
        xinit.cost = cost(xinit, goal)

    # dictionary containing keys 0,1,... for each tree, and values [min, max]
    extreme_costs = {0: [np.inf, 0.], 1: [np.inf, 0.]}
    for i in range(num_props):
        xinit = MG.graphs[i].V[0][0]
        least_cost = min(xinit.cost, extreme_costs[i][0])
        max_cost = max(xinit.cost, extreme_costs[i][1])
        extreme_costs[i] = [least_cost, max_cost]

    global delta_bn, delta_s
    repeat = True
    iter = 0
    while repeat:
        start = timer()
        mc_tot = 0
        for i in xrange(N):
            for p in index_lst:
                i = p[0]
                j = p[1]
                Vac = MG.graphs[i].V[0]
                Vinac = MG.graphs[i].V[1]
                V = Vac + Vinac
                E = MG.graphs[i].E
                S = MG.graphs[i].S
                goal = goallst[j]

                xselected = best_first_selection(V, delta_bn) 

                mc_start = timer()
                t, xnew_traj, u, runcost = MonteCarlo_Prop(xselected, U, Tprop)
                mc_end = timer()
                mc_tot += mc_end - mc_start
                # make a node from final state after propagating
                xnew = Node(xnew_traj[-1])
                xnew.state = fix_state(xnew.state)

                # check velocity constraints
                safe_v = True
                for k in xrange(posdim):
                    cur_v = xnew.state[2*posdim+k]
                    cur_o = xnew.state[3*posdim+k]  # angular velocity
                    if (cur_v < vranges[k][0] or cur_v > vranges[k][1]) or\
                       (cur_o < vranges[k][0] or cur_o > vranges[k][1]):
                        safe_v = False
                if not safe_v:
                    continue

                controlcost = float(np.dot(np.dot(u, R), u))
                timecost = t
                prev_cost_to_go = cost(xselected, goal)
                cost_to_go = cost(xnew, goal)

                # Cost: must tune mult/divfactor, k_T, k_c appropriately
                xnew.cost = xselected.cost - prev_cost_to_go \
                    + cost_to_go + k_T*timecost + k_c*controlcost

                # print("Tree {}".format(i))
                # print("xselected: {}".format(xselected.state[0:3]))
                # print("xnew: {}".format(xnew.state[0:3]))
                # print("goal: {}".format(goal.state[0:3]))
                # print("controlcost: {}, timecost: {}".format(k_c*controlcost, k_T*timecost))
                # print("previous cost_to_go: {}".format(prev_cost_to_go))
                # print("cost_to_go: {}".format(cost_to_go))
                # print("xnew cost: {}".format(xnew.cost))
                # print("")

                # COLLISION (and other) CHECK
                if check_intersect(xselected, xnew, OBS) \
                   and within_boundary(xnew) and safe_v:
                    snear = is_locally_best(xnew, S, delta_s)
                    if snear:
                        xnew.parent = xselected
                        # must update xselected.children as follows or else
                        #  python always changes child list of a single node??
                        xselected.children = xselected.children + [xnew]
                        Vac.append(xnew)
                        E.append(Edge(xnew_traj, xselected, xnew, u, t, runcost))
                        least_cost = min(xnew.cost, extreme_costs[i][0])
                        max_cost = max(xnew.cost, extreme_costs[i][1])
                        extreme_costs[i] = [least_cost, max_cost]
                        prune_nodes(xnew, snear, Vac, Vinac, E)

                for e in pygame.event.get():
                    if e.type == KEYUP and e.key == K_SPACE:
                        repeat = False
                    elif e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                        sys.exit("Leaving because you requested it.")
        iter += N

        # blank screen and update all trees
        screen.fill(white)
        for i in xrange(num_props):
            # display regions again
            border_rect = rects[i]
            colour = colour_lst[i]
            pygame.draw.rect(screen, colour, border_rect)
            pygame.draw.rect(screen, black, border_rect, 1)
            E = MG.graphs[i].E
            least_cost = extreme_costs[i][0]
            max_cost = extreme_costs[i][1]
            for e in E:
                colour = colour_lst[i]
                colour = colour_mult(e.target, colour, least_cost, max_cost)
                pygame.draw.line(screen, map((lambda x: min(x, 255)), colour),
                                 [e.source.state[0], e.source.state[1]],
                                 [e.target.state[0], e.target.state[1]])
        obsDraw(screen, OBS)
        
        # draw solution paths for all appropriate pairs of regions
        endnodes = []
        for p in index_lst:
            i = p[0]
            j = p[1]
            Vac = MG.graphs[i].V[0]  # recall V = [Vac, Vinac]
            Vinac = MG.graphs[i].V[1]
            V = Vac + Vinac
            least_cost = extreme_costs[i][0]
            max_cost = extreme_costs[i][1]
            colour = colour_lst[i]
            endnode = drawSolutionPath(x0lst[i], goallst[j], V, searchradius,
                                       colour, least_cost, max_cost,
                                       pygame, screen)
            endnodes.append(endnode)

        pygame.display.update()

        ##==== SST* - decrease delta_bn and delta_s ====##
        delta_s = epsilon*delta_s
        delta_bn = epsilon*delta_bn 

        end = timer()
        print("Time for iterations {}-{}: {}".format(iter-N, iter-1, end - start))
        print("Time spent computing Monte Carlo: {}".format(mc_tot))
        print("________________________________"*2)
        
        # For saving images

        # pygame.image.save(screen, "sstpoint{:0>2}.bmp".format(iternum))
        #-----------------------------------------------

    main_end = timer()
    print "Total runtime: {}".format(main_end - main_start)
    save_object(MG.graphs, 'MGgraphs_quad_crazy2.pkl')
    save_object(endnodes, 'endnodes_quad_crazy2.pkl')
    return MG, endnodes

if __name__ == '__main__':
    main()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False



