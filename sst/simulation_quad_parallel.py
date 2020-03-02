#!/usr/bin/env python

# Set multiple initial states (one for each ) from which to build trees

import sst_nonlin_quadrotor_4 as sst
from sst_nonlin_quadrotor_4 import MultiGraph, Graph, Node, Edge
from lqr import *
from save_obj import load_object
import numpy as np
from numpy import sin, cos, tan
import pygame
from scipy import integrate
from pygame.locals import *
from numpy.linalg import multi_dot
import colorsys


h = sst.h  # numercial integration timestep
dim = sst.dim
udim = sst.udim
Umax = 15


# #------------------------------------------------------------------------# #

# ####-------------------- Dynamics --------------------#### #

g = -9.8
m = sst.m
Ix = sst.Ix
Iy = sst.Iy
Iz = sst.Iz
J = np.diag([Ix, Iy, Iz])

k_x = np.eye(3)*10
k_v = np.eye(3)*2
k_R = np.eye(3)*2
k_Om = np.eye(3)*2


# init_colour (rgb) transformed to different colour
def animate_colour(init_colour):
    colour = list(colorsys.rgb_to_hsv(*init_colour))
    colour[0] += np.pi/60.
    colour[1] = 0.7
    colour[2] = 0.95
    colour = colorsys.hsv_to_rgb(*colour)
    return np.array(colour)*255


def s(x):
    return np.sin(x)


def c(x):
    return np.cos(x)


# vee map from skew-symmetric matrix to vector
def vee(M):
    x = M[2, 1]
    y = M[0, 2]
    z = M[1, 0]
    arr = np.array([x, y, z])
    return arr


# yaw, pitch, roll. Matrix describes rotation from body-fixed to inertial frame
# Rx * Ry * Rz (?)
def R(ps, th, ph):
    return np.array(
        [[c(th)*c(ps), s(ph)*s(th)*c(ps) - c(ph)*s(ps), c(ph)*s(th)*c(ps) + s(ph)*s(ps)],
         [c(th)*s(ps), s(ph)*s(th)*s(ps) + c(ph)*c(ps), c(ph)*s(th)*s(ps) - s(ph)*c(ps)],
         [-s(th), s(ph)*c(th), c(ph)*c(th)]])


# maps real-valued angle a to the interval [-pi, pi]
def fix_angle(a):
    a = np.mod(a, 2*np.pi)
    if a > np.pi:
        a -= 2*np.pi
    return a


# x is a state, rescales angles using fix_angle function
def fix_state(x):
    a = [fix_angle(xi) for xi in x[3:6]]
    x[3:6] = np.array(a)
    return x


# returns the A matrix (Jacobian) for LQR. Use eqm=0 to NOT approximate
#   with the constant matrix at equilibrium.
def A(t, x, eqm='constant matrix'):
    if eqm == 0:
        x, y, z, ps, th, ph, xdot, ydot, zdot, p, q, r = x
        ps = fix_angle(ps)
        th = fix_angle(th)
        ph = fix_angle(ph)

        # Given are the state values for xdot, ydot, zdot, so we obtain u,v,w
        #   values in the body-fixed frame by multiplying by rotation matrix R.
        # Since R is change of basis from body to inertial frame, this is
        #   defined to mean:   e_i = Rinv b_i, so in this case we want
        #   b_i = R e_i
        U = np.dot(R(ps, th, ph), np.array([[xdot, ydot, zdot]]).T)
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
        A[6:8, 4:6] = np.array([[g, 0], [0, -g]])

    return A

B = np.zeros([dim, udim])
B[9:12, 1:4] = np.array([[1./Ix, 0, 0], [0, 1./Iy, 0], [0, 0, 1./Iz]])
B[8, 0] = 1./m

# Q = np.eye(sst.dim)*1
# Q[0:3, 0:3] = np.eye(3)*3  # position weight
# Q[3:6, 3:6] = np.eye(3)*1  # angle weight
# Q[6:9, 6:9] = np.eye(3)*4 # translational velocity weight
# Q[9:12, 9:12] = np.eye(3)*6  # rotational velocity weight

# R_lqr = np.eye(udim)*20  # control cost matrix R appearing in product (u.T,R,u)

Q = np.eye(sst.dim)*1
Q[0:3, 0:3] = np.eye(3)*2  # position weight
Q[3:6, 3:6] = np.eye(3)*1  # angle weight
Q[6:9, 6:9] = np.eye(3)*2  # translational velocity weight
Q[9:12, 9:12] = np.eye(3)*3  # rotational velocity weight

R_lqr = np.eye(udim)*5e10  # control cost matrix R appearing in product (u.T,R,u)
R_lqr[0, 0] = 1

# #------------------------------------------------------------------------# #

"""
state x = [[x, y, z, psi, theta, phi, xdot, ydot, zdot, p, q, r]].T
  where:
    x, y, z          -- position (World frame)
    psi, theta, phi  -- yaw, pitch, roll  (World frame)
    xdot, ydot, zdot -- translational velocity (World frame)
    p, q, r          -- angular velocities (Body frame)

    u, v, w          -- translational velocity (Body frame)

    Note that we approximate [phidot, thetadot, psidot] == [p, q, r]


"""


def fcn(t, x, u):
    x = fix_state(x)
    g17 = -1./m * ( sin(x[5])*sin(x[3]) + cos(x[5])*cos(x[3])*sin(x[4]) )
    g18 = -1./m * ( sin(x[5])*cos(x[3]) - cos(x[5])*sin(x[3])*sin(x[4]) )
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


# #-------------------------------------------------------------------------# #
# Obtain list of edges of candidate trajectory given endnode, graph G, xinit
def construct_edgelist(endnode, G, xinit):
    edgelst = []
    timecost = 0
    curnode = endnode
    while not(np.array_equal(curnode.state, xinit.state)):
        for e in G.E:
            if np.array_equal(e.target.state, curnode.state):
                edgelst.append(e)
                timecost += e.time
                curnode = e.source
                break
    edgelst.reverse()
    return edgelst, timecost


# simulate trajectory starting at x and along the edges of edgelst
def simulate(x, edgelst, screen):
    colour = [255, 246, 0]
    P = np.eye(12)  # P0 used to integrate ricatti eqn
    for e in edgelst:
        # *** uncomment this next line when using file: MGgraphs_quad.pkl
        # e.traj = [x.state] + e.traj  # this is a fix for bad saved file.
        x.state = fix_state(x.state)
        T = e.time
        tmp = x

        ode = integrate.ode(fcn, A)  # A used as Jacobian matrix
        ode.set_initial_value(x.state, 0)
        # ode.set_jac_params(0)  # change/remove this for constant A matrix as Jac
        ode.set_integrator('vode', nsteps=2000, method='bdf')
        newtraj = [x.state]
        ts = []
        iternum = 0
        while ode.successful() and ode.t < T:
            x.state = fix_state(x.state)
            e.traj[iternum] = fix_state(e.traj[iternum+1])

            R_B = R(*x.state[3:6])
            R_D = R(*e.traj[iternum][3:6])
            Om_B = x.state[9:12]
            Om_D = e.traj[iternum][9:12]

            err = np.array(x.state) - np.array(e.traj[iternum])
            err[3:6] = 1/2 * vee(np.dot(R_B.T, R_D) - np.dot(R_D.T, R_B))
            err[9:12] = multi_dot([R_B.T, R_D, Om_D]) - Om_B

            print("_______________"*3)

            print("|error| = {}".format(np.linalg.norm(err)))

            print("    actual x pos: {}".format(x.state[0:3]))
            print("desired traj pos: {}".format(e.traj[iternum][0:3]))
            print("")
            print("     pos error: {}".format(err[0:3]))
            print("   angle error: {}".format(err[3:6]))
            print("    velo error: {}".format(err[6:9]))
            print("ang velo error: {}".format(err[9:12]))
            print("")

            # LQR
            K, P = full_feedback(A(ode.t, x.state), B, Q, R_lqr, P, ode.t, h)
            LQR_term = np.array(np.dot(-K, err))
            # LQR_term = np.zeros(4)
            u =  e.control + LQR_term
            # u = LQR_term
            print("Orig control: {}\nLQR adjust: {}\nResulting control: {}".format(e.control, LQR_term, u))

            # # --- trajectory smoothing --- #

            # # delta = (2*M*(N+1) - beta*(M-1))/(3*M+1)  # new estimate
            # # delta = (M*(N+1) - 2*beta*(M-1))/(M+1)  # old
            # # delta = int(np.ceil(delta))-1
            # delta = int((N+1)/2)

            # safe = False  # in order to enter the loop and begin smoothing
            # while not(safe):
            #     safe = True
            #     all_xs, all_ys, all_zs = [[], [], []]
            #     P = {}  # dictionary containing associated poly coeffs for x, y, z, yaw
            #     for var in ['x', 'y', 'z', 'yaw']:
            #         d = create_dvec(M, delta, waypoints, var)
            #         p, newd = smooth(M, N, beta, delta, T, d)
            #         P[var] = p

            #     for m in range(M):
            #         tt = np.linspace(0, T[m], num=NUM)
            #         xs = polyval(tt, P['x'][m*(N+1):(m+1)*(N+1)]).reshape(NUM,)
            #         ys = polyval(tt, P['y'][m*(N+1):(m+1)*(N+1)]).reshape(NUM,)
            #         zs = polyval(tt, P['z'][m*(N+1):(m+1)*(N+1)]).reshape(NUM,)
            #         # yaws = polyval(tt, P['yaw'][m*(N+1):(m+1)*(N+1)]).reshape(NUM,)
            #         all_xs.extend(xs); all_ys.extend(ys); all_zs.extend(zs)
            #         safe_m = True
            #         for i in range(NUM-1):
            #             within_bounds = within_boundary([xs[i], ys[i], zs[i]])
            #             no_collision = collision_check_3d([xs[i], ys[i], zs[i]],
            #                                               [xs[i+1], ys[i+1], zs[i+1]], OBS)
            #             safe_m = safe_m and within_bounds and no_collision
            #             if not(within_bounds and no_collision):
            #                 print("Warning at step m =", m, ", i =", i)
            #                 if not within_bounds:
            #                     print("Not within bounds")
            #                 if not no_collision:
            #                     print("Collision with obstacle")
            #                     ax.scatter([xs[i]], [ys[i]], [zs[i]], c='k', s=30)
            #         if not(safe_m):
            #             safe = False
            #             x0 = waypoints[m].state
            #             x1 = waypoints[m+1].state
            #             tau = T[m]
            #             t = tau/2.
            #             newstate = ansol.x(x0, x1, t, tau)
            #             newNode = Node(newstate)
            #             print("new state added at step m={}: {}".format(m, newstate))
            #             waypoints = waypoints[:m+1] + [newNode] + waypoints[m+1:]
            #             T = T[:m] + [t] + [t] + T[m+1:]
            #             M += 1

            # limit possible u values?
            # for ind in xrange(sst.udim):
            #     if ind == 0 and abs(u[ind]) > 50:
            #         u[ind] = u[ind]/abs(u[ind]) * 50
            #     if ind > 0 and abs(u[ind]) > Umax:
            #         u[ind] = u[ind]/abs(u[ind]) * Umax

            ode.set_f_params(u)
            ode.integrate(ode.t + h)
            ts.append(ode.t)
            newtraj.append(ode.y)
            iternum += 1
            x = sst.Node(ode.y) # proceed from latest integrated state
            # print "u: {}".format(u)
            # print "x:", x.state
        try:
            pygame.draw.line(screen, colour,
                     [tmp.state[0], tmp.state[1]], [x.state[0], x.state[1]], 4)
            colour = animate_colour(colour)
        except:
            pygame.display.update()
            print('Breaking -- Error: failed to draw line in pygame')
            break
        if np.any(x.state[0:3] > 1500) or np.any(x.state[0:3] < 0):
            print('Breaking -- Error: state position exceeds screen size')
            break

    pygame.display.update()
    return x


# #--------------------------------# #
def main():
    # Initialization
    MG = sst.MGinit
    MG.graphs = load_object('MGgraphs_quad_crazy.pkl')
    endnodes = load_object('endnodes_quad_crazy.pkl')

    pygame.init()
    screen = pygame.display.set_mode(sst.WINSIZE)
    pygame.display.set_caption('SST')
    white = 255, 240, 240
    black = 20, 20, 40
    green = 50, 150, 50
    screen.fill(white)
    rects = []  # list of rectangles for redrawing
    colour_lst = sst.colour_lst

    rects = []  # list of rectangles for redrawing
    for i in range(sst.num_props):
        wh = sst.regionsize_list[i]  # pair (width, height)
        colour = colour_lst[i]
        region = sst.init_coord_list[i][:2]  # x, y values from init state [i]
        # adjust so goal in centre of region by subtracting half of wh[i]
        border_rect = pygame.Rect(region[0] - wh[0]/2., region[1] - wh[1]/2.,
                                  wh[0], wh[1])
        rects.append(border_rect)
        pygame.draw.rect(screen, colour, border_rect)
        pygame.draw.rect(screen, black, border_rect, 1)

    sst.obsDraw(screen, sst.OBS)
    pygame.display.update()

    # draw candidate trajectories, and create edgelst for each traj

    idx = 0
    edgelsts = []
    costs = []
    index_lst = sst.goal_index_lst(sst.num_props)
    for p in index_lst:
        i = p[0]
        j = p[1]
        curnode = endnodes[idx]
        x0 = MG.graphs[i].V[0][0]
        colour = colour_lst[i]
        while not(np.array_equal(curnode.state, x0.state)):
            pygame.draw.line(screen, colour*0.8,
                             [curnode.state[0], curnode.state[1]],
                             [curnode.parent.state[0], curnode.parent.state[1]], 7)
            curnode = curnode.parent
        newlst, timecost = construct_edgelist(endnodes[idx], MG.graphs[i], x0)
        edgelsts.append(newlst)
        costs.append(timecost)
        idx += 1

    pygame.display.update()

    # ##-------------------------------## #
    # Run simulation

    x = MG.graphs[0].V[0][0]

    count = 2
    while count > 0:
        x = simulate(x, edgelsts[0] + edgelsts[1], screen)
        count -= 1


if __name__ == '__main__':
    main()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
