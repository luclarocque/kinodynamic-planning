import numpy as np
from functools import reduce
from numpy.linalg import multi_dot
from numpy.polynomial.polynomial import polyval, polyder
from pprint import pprint
# from fmt_quad import *

np.set_printoptions(suppress=True)

# This version of smoother assumes that
#   of the unknown derivatives, we only need to find half of them
#   since the rest are given by the continuity constraints 

# tuple (state1, state2, opt_cost)
# traj = get_traj()

M = 3  # number of segments
# N = 3
# N = 5
N = 9  # order of polynomials -- should be odd?
beta = 1  # number of known derivatives of intermediate waypoints

# T = 3/4*np.ones(M)  # list of optimal times
T = [0.75, 1.2, 1.1]
w = 0.  # placeholder for unknown derivatives. 
        # MUST BE 0 since we use this to impose continuity: d_T - d_0 = w = 0

# d is a concatenated vector where, for each segment, the vector the form:
#  [di_0_0, di_0_1, ..., di_0_(delta-1), di_T_0, di_T_1, ..., di_T_(delta-1)]
# d = np.array([[10., 0., 11., w, 11., w, 12., 0.]]).T
# d = np.array([[10., 0., 0.,  11., w, w, 
#                11., w, w, 12., 0., 0.]]).T
d = np.array([[10., 0., 0., 0., 0.,  
               11., w, w, w, w,
               11., w, w, w, w,
               12., w, w, w, w,
               12., w, w, w, w,
               13., 0., 0., 0., 0.]]).T


# returns n!
def factorial(n):
    return reduce(lambda x,y:x*y,[1]+list(range(1,n+1)))


def permutation_matrix(M, beta, delta):
    # begin by building ordering vector, e.g., np.array([0,1,2,3,5,7,8,9,4,6])
    order_vec = list(np.zeros(2*delta*M))
    idx = 0

    # know the first delta derivs (first segment)
    for i in range(delta):  
        order_vec[idx] = i
        idx += 1
    
    # for every intermediate block of 2*delta derivs, we can assume to know the
    #   first delta of them -> will be equal to next delta derivs we optimize
    constraint_lst = []
    for m in range(2*(M-1)):
        if m % 2 == 0:
            for i in range(delta):
                order_vec[idx] = delta + m*delta + i
                if i>=beta:
                    constraint_lst.append(delta + m*delta + i)
                idx += 1
        else:
            for b in range(beta):
                order_vec[idx] = delta + m*delta + b
                idx += 1
        

    # know the last delta derivs (last segment)
    for i in range(2*delta*M - delta, 2*delta*M):  
        order_vec[idx] = i
        idx += 1

    S = set(list(range(2*delta*M)))
    S_fix = set(order_vec)
    S_free = S - S_fix
    sorted_free = sorted(list(S_free))
    n_free = len(sorted_free)
    n_fix = 2*delta*M - n_free
    for i in range(n_free):
        order_vec[idx] = sorted_free[i]
        idx += 1
    # print("order_vec:\n", order_vec)
    # print("constraint_lst:\n", constraint_lst)

    # create permutation matrix C using order_vec
    C = np.eye(2*delta*M)
    idmat = np.eye(2*delta*M)
    for i in range(2*delta*M):
        C[i] = idmat[order_vec[i]]

    # add continuity constraints in new matrix CC
    CC = C.copy()

    for c in constraint_lst:
        idx = order_vec.index(c)
        CC[idx] = CC[idx] - idmat[c + delta]

    return CC, n_free


def compute_yaw(curstate):
    if curstate[3] == 0:
        if curstate[4] == 0:
            yaw = 0.
        else:
            yaw = np.arctan(curstate[4]*np.inf)
    else:
        yaw = np.arctan(curstate[4]/curstate[3])  # arctan(ydot/xdot)
    return yaw


# initializes the vector of derivatives, d, given a list of waypoints (Nodes),
#   and state_var which is one of {'x', 'y', 'z', 'yaw'}
def create_dvec(M, delta, waypoints, state_var):
    dim = 2*delta*M
    d = np.zeros((dim, 1))
    vardict = {'x': 0, 'y' : 1, 'z' : 2, 'yaw' : 9}  # 9 used as placeholder

    state_ind = vardict[state_var]
    if state_ind in [0,1,2]:
        curstate = waypoints[0].state
        d[0] = curstate[state_ind]
        for m in range(1, 2*M-1):
            mm = int(np.ceil(m/2))  # if odd, stays same, otherwise subtracts 1
            curstate = waypoints[mm].state
            d[m*delta] = curstate[state_ind]
        curstate = waypoints[-1].state
        d[-delta] = curstate[state_ind]
    else:
        curstate = waypoints[0].state
        d[0] = compute_yaw(curstate)
        for m in range(1, 2*M-1):
            mm = int(np.ceil(m/2))  # if odd, stays same, otherwise subtracts 1
            curstate = waypoints[mm].state
            d[m*delta] = compute_yaw(curstate)
        curstate = waypoints[-1].state
        d[-delta] = compute_yaw(curstate)
    return d



# returns the polynomial coefficients in a list, and new d vector 
#   with optimized derivatives
def smooth(M, N, beta, delta, T, d):
    Q = np.zeros(( M*(N+1), M*(N+1) ))
    A = np.zeros(( 2*delta*M, M*(N+1) ))
    for m in range(M):
        A_0 = np.zeros(( delta, N+1 ))
        A_T = np.zeros(( delta, N+1 ))
        for i in range(N+1):
            for j in range(N+1):
                if i >= 4 and j >= 4:
                    prod = factorial(i)*factorial(j)/(factorial(i-4)*factorial(j-4))
                    Q[m*(N+1) + i, m*(N+1) + j] = 2*prod/(i+j-7)* T[m]**(i+j-7)
                if i == j and i < delta:
                    prod = factorial(i)
                    A_0[i, j] = prod
                if j >= i and i < delta:
                    prod = factorial(j)/factorial(j-i)
                    A_T[i, j] = prod * T[m]**(j-i)
        A[2*delta*m : 2*delta*(m+1), 
            m*(N+1) : (m+1)*(N+1)] = np.append(A_0, A_T, 0)

    C, n_free = permutation_matrix(M, beta, delta)
    n_fix = 2*delta*M - n_free

    # Q = np.eye(M*(N+1))  # REMOVE!!! This is for testing only

    Ainv = np.linalg.pinv(A)
    Cinv = np.linalg.inv(C)
    H = multi_dot([Cinv.T, Ainv.T, Q, Ainv, Cinv])
    H11 = H[:n_fix, :n_fix]
    H12 = H[:n_fix, n_fix:]
    H21 = H[n_fix:, :n_fix]
    H22 = H[n_fix:, n_fix:]

    Cd = multi_dot([C, d])  # [d_fix  d_free].T

    d_free_opt = multi_dot([-np.linalg.inv(H22), H12.T, Cd[:n_fix]])
    Cd[n_fix:] = d_free_opt
    newd = multi_dot([Cinv, Cd])

    p = multi_dot([Ainv, newd])  # concatenated vector of coeffs
    # print("poly coeffs, N={}, M={}\n".format(N, M), p)

    return p, newd

if __name__ == '__main__':
    delta = (2*M*(N+1) - beta*(M-1))/(3*M+1)
    # delta = (M*(N+1) - 2*beta*(M-1))/(M+1)
    delta = int(np.ceil(delta))-1
    print("delta: {}".format(delta))

 
    # waypoints = [Node((0.5,0.6,0.7,0,0,0)), Node((1,1,1,1,1,1)), Node((2,2,2,3,4,5)), Node((3,3,3,1,np.sqrt(3),3))]
    # d = create_dvec(3, delta, waypoints, 'yaw')
    # print(d)

    Plst, newd = smooth(M, N, beta, delta, T, d)
    print("resultant d:\n", newd, "\n")

    for i in range(M):
        print("position poly"+str(i)+": ", polyval([0., T[i]], Plst[i*(N+1):(i+1)*(N+1)] ))
        print("velocity poly"+str(i)+": ", polyval([0., T[i]], polyder(Plst[i*(N+1):(i+1)*(N+1)]) ))
        print()
