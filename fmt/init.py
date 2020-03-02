# Initial step for performing FMT planning.\
# This module saves:
#   - the set of sampled nodes, V,
#   - the set of sampled pairs, sample_pairs,
#   - the roadmap dictionary (state0, state1):(J_opt, tau), called COST
#   - the SVM classifier, classifier

# from fmt_quad import *
import sys
from fmt_quad import cost, reachable_set
from save_obj import save_object, load_object
from trainsvm import trainsvm
from analytic_solver_doubleint import Jstar, find_tau
from sampler import sample, sample_data

C = 0.05

# set the size of the workspace
XDIM = 10.
YDIM = 10.
ZDIM = 10.


# Computes v.N_in and v.N_out for all nodes v given by fname_V (filename or set)
def precompute_reachable_sets(fname_V, fname_COST, fname_classifier, Jth):
    print("Memoizing reachable sets...")
    sys.setrecursionlimit(200000)
    if isinstance(fname_V, str):
        V = load_object(fname_V)
    else:
        V = fname_V
    if isinstance(fname_COST, str):
        COST = load_object(fname_COST)
    else:
        COST = fname_COST
    if isinstance(fname_classifier, str):
        classifier = load_object(fname_classifier)
    else:
        classifier = fname_classifier

    N = len(V)
    i=1
    for v in V:
        print("  Updating node {} / {}".format(i, N))
        v.N_out = reachable_set(v, V, Jth, COST, classifier)
        v.N_in = reachable_set(v, V, Jth, COST, classifier, forward=False)
        i+=1
    return V





def main(n, Jth):
    # V = set([xinit])
    V = set()

    print("Sampling...")
    COST = dict()
    V |= sample(n, (0., 0., 0.), XDIM, YDIM, ZDIM, COST, ax=None)

    # sample_pairs: a set with elements (state0, state1)
    Npair = int(n*(n-1)/20)
    sample_pairs = sample_data(V, Npair)

    ## --- COST roadmap --- ##

    # create a dictionary COST with entries:  (state0, state1) : (opt_cost, opt_time) 
    print("Constructing (sampled) COST roadmap...")
    for p in sample_pairs:
        tau = find_tau(*p)
        J_opt = Jstar(*p, tau)
        COST[p] = (J_opt, tau)

    # create svm.SVC object which classifies pairs of states with C and Jth
    classifier = trainsvm(COST, C, Jth)

    # add every possible edge to COST
    print("Completing COST roadmap...")
    sys.setrecursionlimit(2*n*n)
    N = len(V)*(len(V)-1)
    i = 1
    for u in V:
        for v in V - set([u]):
            if i % 1000 == 1:
                print("  Processing {} / {}".format(i, N))
            cost(u, v, COST)  # Computes and stores cost in COST
            if i % 50000 == 0:
                save_object(COST, 'COST_full_n{}_J{}.pkl'.format(int(n), int(Jth)))
            i += 1

    V = precompute_reachable_sets(V, COST, classifier, Jth)

    print("Saving objects.")
    save_object(V, 'V_n{}_J{}.pkl'.format(int(n), int(Jth)))
    save_object(COST, 'COST_full_n{}_J{}.pkl'.format(int(n), int(Jth)))
    save_object(classifier, 'classifier{}node_J{}.pkl'.format(n, int(Jth)))
    print("Success! n: {}, Jth: {}".format(n, int(Jth)))



if __name__ == "__main__":
    n = 250
    Jth = 250
    main(n, Jth)