import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from line_intersect import collision_check_3d
from analytic_solver_doubleint import Jstar, find_tau
from plot_3d import within_boundary, goal_intervals, draw_3d
from smoother import smooth_traj
from sampler import sample
from numpy.polynomial.polynomial import polyval
from classes import Node, Edge
import save_obj
# import timing
import time
import sys

###############################################################################





###############################################################################

N = 9  # degree of polynomials for trajectory smoothing
beta = 1  # number of known derivs at intermediate waypoints
NUM = 50  # number of points to plot for each smoothed segment
n_goal = 0  # number of samples to take from goal region

# the format for 3d objects (rectangular prisms) is:
#   [corner position (triple), width, depth, height]
# OBS = [[(1, 4, 1), 1, 6, 5],
#        [(3, 4, 3), 5, 1, 5]]
# OBS = [[(4, 4, 2), 2, 2, 7]]
OBS = []

XDIM, YDIM, ZDIM = [10, 10, 10]
DIMsize = [XDIM, YDIM, ZDIM]
screen_intervals = [(0, XDIM), (0, YDIM), (0, ZDIM)]


# optimal cost from node n1 to n2.
def cost(n1, n2, COST):
    p = (n1.state, n2.state)
    if p in COST:
        return COST[p][0]
    else:
        tau = find_tau(*p)
        J_opt = Jstar(*p, tau)
        COST[p] = (J_opt, tau)
        return J_opt

# plot solution path from xinit to endnode, and return waypoints as list
def draw_path(xinit, endnode, ax, plot_waypoints=True):
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
        zs = (source.state[2], target.state[2])
        if plot_waypoints:
            ax.plot(xs, ys, zs, lw=2, color='b', zorder=1)
        target = source
        source = source.parent
    return list(deq)


# finds the nodes of S that are within cost Jth of node x using
#   a trained svm classifier and the COST
# Finds forward-reachable set unless forward=False, in which case
#   the backward reachable set is computed (i.e., for computing x.N_in)
def reachable_set(x, S, Jth, COST, classifier, forward=True):
    if forward is True:
        if x.N_out is None:
            # print("forward reachable set not found.")
            S = S - set([x])
            x.N_out = set()
            for s in S:
                p = (x.state, s.state)
                if p in COST:
                    if COST[p][0] < Jth:
                        x.N_out |= {s}
                else:
                    p = np.append(x.state, s.state)
                    prediction = classifier.predict([p])
                    # print("Forward: classifier found: {}".format(prediction))
                    if prediction == 1:
                        x.N_out |= {s}
            return x.N_out
        else:
            return x.N_out

    if forward is False:
        if x.N_in is None:
            S = S - set([x])
            x.N_in = set()
            for s in S:
                p = (s.state, x.state)
                if p in COST: 
                    if COST[p][0] < Jth:
                        x.N_in |= {s}
                else:
                    p = np.append(s.state, x.state)
                    prediction = classifier.predict([p])
                    # print("Backward: classifier found: {}".format(prediction))
                    if prediction == 1:
                        x.N_in |= {s}
            return x.N_in
        else:
            return x.N_in


###############################################################################

def planner(n, Jth, xinit, Xgoal, ax,
            plot_bool=False, all_lines=False, 
            plot_waypts=True, plot_smooth=True, no_smoothing=False):
    # --- setup environment --- #
    sys.setrecursionlimit(10*n*(n-1))

    Xgoal_pos = Xgoal[0]
    Xgoal_width = Xgoal[1]
    Xgoal_depth = Xgoal[2]
    Xgoal_height = Xgoal[3]
    g_intervals = goal_intervals(Xgoal)
    for o in OBS:
        draw_3d(*o, ax, color=(0.8, 0.2, 0.2, 1.))
    
    # --------------------------- #
    #     kinoFMT algorithm       #
    # --------------------------- #

    # --- initialize graph ---------------------------------------------------- #
    # reinitialize xinit to ensure it starts "fresh"
    xinit = Node(xinit.state)
    # ax.scatter(*xinit.state[:3], s=64, marker='*')  # plot initial state

    # load the presampled set of states for this (n, Jth) configuration, and
    #   add xinit to this set.
    V = save_obj.load_object('V_n{}_J{}.pkl'.format(int(n),int(Jth))) | {xinit}
    # COST roadmap (dictionary)
    COST = save_obj.load_object('COST_full_n{}_J{}.pkl'.format(int(n), int(Jth)))
    # classifier is a svm.SVC object which classifies a pair of states with Jth
    classifier = \
        save_obj.load_object('classifier{}node_J{}.pkl'.format(n, int(Jth)))
    
    #--------------------------------------------------------------------------
    # Remove any nodes that happen to lie in the goal region -- we only
    #   want to deal with goal nodes we sample from goal explicitly.
    #--------------------------------------------------------------------------
    # goal_set = set()
    # for v in V - {xinit}:
    #     within_goal = within_boundary(v, g_intervals)
    #     if within_goal:
    #         goal_set |= {v}
    #         print("goal state:", v.state)
    # V -= goal_set
    # print("Total removed from goal:", len(goal_set))
    #
    # # FOR DEBUGGING
    # tot_pred = 0
    # for q in goal_set:
    #     for v in V - {q}:
    #         p = np.append(v.state, q.state)
    #         pred = classifier.predict([p])
    #         if pred: tot_pred += 1
    # oldgoal = goal_set.pop()
    # goal_set |= {oldgoal}
    # print("Total predicted in N_in for removed goal state:", tot_pred)
    # print("N_in actual:", len(oldgoal.N_in))
    # print("iself?",oldgoal in oldgoal.N_in)
    # print("cost:", oldgoal.cost)
    # for q in goal_set:
    #     print("Removed:", q.state)
    #     for v in V - {xinit}:
    #         if q in v.N_out:
    #             v.N_out -= {q}
    #         if q in v.N_in:
    #             v.N_in -= {q}
    #--------------------------------------------------------------------------

    E = set([])

    H = {xinit}  # frontier of exploration
    W = V - {xinit}  # copy of V setminus {xinit} (unvisited)
    z = xinit  # point in the frontier, H

    stime = time.time()

    # find xinit reachable set
    V |= {xinit}
    xinit.N_out = reachable_set(xinit, V, Jth, COST, classifier)
    for v in xinit.N_out:
        # cost(xinit, v, COST)  # add cost to COST dictionary
        v.N_in |= {xinit}

    # find Xgoal reachable sets
    Vgoal = sample(n_goal, Xgoal_pos, Xgoal_width, Xgoal_depth, Xgoal_height, 
                   COST, ax=None, zero_vel=True)
    # # FOR DEBUGGING
    # for q in goal_set:
    #     replacement = Node(q.state)
    #     Vgoal |= {replacement}
    #     print("Added replacement:", replacement.state)
    for vg in Vgoal:
        vg.N_in = reachable_set(vg, V, Jth, COST, classifier, forward=False)
        print("goal state N_in contains", len(vg.N_in))
        # add vg to the outgoing nbhd of all nodes that can reach it
        for q in vg.N_in:
            # cost(q, vg, COST)  # add cost to COST dictionary
            q.N_out |= {vg}
    V |= Vgoal

    print("Running kinoFMT...")
    kinotimer = time.time()
    coltimer = 0. # accumulates collision-check time
    i = 0
    while not(within_boundary(z, g_intervals)):
        N_zout = reachable_set(z, V, Jth, COST, classifier)
        Xnear = N_zout & W  # set of as yet unreached nodes in reachable set
        for x in Xnear:
            # find backward reachable set of x
            N_xin = reachable_set(x, V, Jth, COST, classifier, forward=False)
            Ynear = N_xin & H  # set of frontier nodes that can reach x
            # find ymin if it exists, otherwise proceed to next x.
            if len(Ynear) == 0:
                continue
            ymin = random.choice(tuple(Ynear))  # initialize with random y
            ymin_x_cost = ymin.cost + cost(ymin, x, COST)
            for y in Ynear:
                y_x_cost = y.cost + cost(y, x, COST)
                if y_x_cost < ymin_x_cost:
                    ymin = y
                    ymin_x_cost = y_x_cost

            colcurtime = time.time()
            no_collision = collision_check_3d(ymin, x, OBS)
            coltimer += time.time() - colcurtime

            if no_collision:
                # ymin.children |= {x}
                x.parent = ymin
                x.cost = ymin_x_cost
                E |= set([Edge(ymin, x)])
                H |= set([x])
                W -= set([x])
                xs = (ymin.state[0], x.state[0])
                ys = (ymin.state[1], x.state[1])
                zs = (ymin.state[2], x.state[2])
                if all_lines:
                    ax.plot(xs, ys, zs, lw=1, zorder=1)
        H -= {z}
        if len(H) == 0:
            print("Failure")
            return -1, [], -1  # so as to not break when returning P,T,z
        # find argmin h in H of cost-to-come
        z = random.choice(tuple(H))
        for h in H:
            if h.cost < z.cost:
                z = h
        i += 1
    kinotime = time.time() - kinotimer

    # plt.axis('scaled')
    draw_3d(*Xgoal, ax)
    waypoints = draw_path(xinit, z, ax, plot_waypoints=plot_waypts)  # list of nodes from start to finish
    
    # create list T of optimal times for each segment/edge
    M = len(waypoints)-1
    T = []
    for m in range(M):
        n1 = waypoints[m]
        n2 = waypoints[m+1]
        p = (n1.state, n2.state)
        t = COST[p][1]
        T.append(t)
    print("M =", M)
    print("T =", T)

    if no_smoothing:
        if plot_bool:
            plt.show()
        return waypoints, T, z

    # --- trajectory smoothing --- #
    smoothtimer = time.time()
    print("")
    P, all_xs, all_ys, all_zs = smooth_traj(waypoints, T, N, beta,
                                            OBS, screen_intervals, NUM)

    print("")
    print("Time spent smoothing:", time.time() - smoothtimer)
    print("Time spent running kinoFMT:", kinotime)
    print("Total collision_check_3d computation time: {}\n".format(coltimer))
    print("Actual execution time: {}".format(time.time() - stime))
    print("-----"*4)
    
    if plot_smooth:
        ax.plot(all_xs, all_ys, all_zs, lw=4, color='g', zorder=1)#, linestyle='--')
    if plot_bool:
        # plt.axis('scaled')
        plt.show()

    return P, T, z

if __name__ == "__main__":
    # Be sure (n, Jth) files were pre-initialized with init.py
    n = 200
    Jth = 260

    # xinit = Node((1., 1., 1., 0., 0., 0.))
    # xinit = Node((1.5, 1., 8., 0., 0., 0.))
    xinit = Node((1.5, 1., 0., 0., 0., 0.))
    Xgoal = [(6., 9., 8.), 1., 1., 1.]
    # Xgoal = [(6., 9.5, 8.), 1., 0.5, 1.]
    # Xgoal = [(1., 0., 8.), 1., 1., 1.]
    # Xgoal = [(9., 1., 2.5), 1., 1., 1.]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_xlim3d(0, 10)
    ax.set_ylabel('Y')
    ax.set_ylim3d(0, 10)
    ax.set_zlabel('Z')
    ax.set_zlim3d(0, 10)

    planner(n, Jth, xinit, Xgoal, ax, plot_bool=True, all_lines=False,
            plot_waypts=False, plot_smooth=True, no_smoothing=False)


    #-----------------------------------------------------------
    # Use the following for plotting multiple runs
    #-----------------------------------------------------------
    
    # xinit = Node((1.5, 1., 0., 0., 0., 0.))
    # Xgoal = [(6., 9., 8.), 1, 1, 2]

    # fig = plt.figure()
    # smoothing = False

    # #-----------------------------------------------------------
    # n = 200
    # Jth = 300

    # ax = fig.add_subplot(221, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_xlim3d(0, 10)
    # ax.set_ylabel('Y')
    # ax.set_ylim3d(0, 10)
    # ax.set_zlabel('Z')
    # ax.set_zlim3d(0, 10)

    # planner(n, Jth, xinit, Xgoal, ax, all_lines=True, plot_smooth=smoothing)
    # #-----------------------------------------------------------
    # n = 200
    # Jth = 260

    # ax = fig.add_subplot(222, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_xlim3d(0, 10)
    # ax.set_ylabel('Y')
    # ax.set_ylim3d(0, 10)
    # ax.set_zlabel('Z')
    # ax.set_zlim3d(0, 10)

    # planner(n, Jth, xinit, Xgoal, ax, all_lines=True, plot_smooth=smoothing)
    # #-----------------------------------------------------------
    # n = 1000
    # Jth = 200

    # ax = fig.add_subplot(223, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_xlim3d(0, 10)
    # ax.set_ylabel('Y')
    # ax.set_ylim3d(0, 10)
    # ax.set_zlabel('Z')
    # ax.set_zlim3d(0, 10)

    # planner(n, Jth, xinit, Xgoal, ax, all_lines=True, plot_smooth=smoothing)
    # #-----------------------------------------------------------
    # n = 2000
    # Jth = 150

    # ax = fig.add_subplot(224, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_xlim3d(0, 10)
    # ax.set_ylabel('Y')
    # ax.set_ylim3d(0, 10)
    # ax.set_zlabel('Z')
    # ax.set_zlim3d(0, 10)

    # planner(n, Jth, xinit, Xgoal, ax, all_lines=True, plot_bool=True, plot_smooth=smoothing)


    # fig.savefig('figure_smooth.png', bbox_inches='tight')
