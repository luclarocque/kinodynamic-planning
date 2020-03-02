import numpy as np
import scipy.linalg
from scipy import integrate
from functools import reduce
from numpy.linalg import multi_dot


# Compute LQR gain
def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    P = np.array(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.array(multi_dot([scipy.linalg.inv(R), B.T, P]))

    eigVals, eigVecs = scipy.linalg.eig(A - multi_dot([B, K]))

    return K, P, eigVals


# The function to integrate forward (used in full_feedback)
def diff_riccati(t, P, A, B, Q, R):
    P = np.reshape(P, A.shape)
    dPdt = -Q - multi_dot([A.T, P]) - multi_dot([P, A]) \
            + multi_dot([P, B, np.linalg.inv(R), B.T, P])
    dPdt = np.reshape(dPdt, (A.size, 1))
    return dPdt


# returns K(t) to use for LQR feedback. t needs to be a linspace.
def full_feedback(A, B, Q, R, P0, t, h):
    # t = np.linspace(t, t+h, num=5)
    P0 = np.reshape(P0, (P0.size, 1))

    ode = integrate.ode(diff_riccati)
    ode.set_initial_value(P0, t)
    ode.set_integrator('dopri5', nsteps=50000, method='bdf')
    ode.set_f_params(A, B, Q, R)
    ode.integrate(ode.t + h)
    newP = ode.y


    # newP = integrate.odeint(diff_riccati, P0, t, (A, B, Q, R), mxstep=1000)
    # newP = newP[-1]

    newP = np.reshape(newP, A.shape)  # (A.shape[0], A.shape[1])
    # print "newP:\n", newP
    K = multi_dot([np.linalg.inv(R), B.T, newP])
    eigVals, eigVecs = scipy.linalg.eig(A - multi_dot([B, K]))
    return K, newP, eigVals


if __name__ == '__main__':
    dim = 12
    udim = 4
    m = 0.5
    g = 9.8
    Ix = 0.3
    Iy = 0.3
    Iz = 0.3

    A = np.zeros([dim, dim])
    A[0:3, 6:9] = np.eye(3)
    A[3:6, 9:12] = np.fliplr(np.eye(3))
    A[6:8, 4:6] = np.array([[-g, 0], [0, g]])
    # A = np.eye(dim)

    B = np.zeros([dim, udim])
    B[9:12, 1:4] = np.array([[1./Ix, 0, 0], [0, 1./Iy, 0], [0, 0, 1./Iz]])
    B[8, 0] = -1./m
    # B = np.zeros((dim, udim))
    # B[0:3,0:3] = np.eye(3)    # NOTE: TODO: B seems to be breaking the integration

    Q = np.eye(dim)*1

    R = np.eye(udim)*10

    P0 = np.eye(dim)*1
    # P0 = np.ones((dim, dim))
    t = 1.57
    h = 0.1

    K, P = full_feedback(A, B, Q, R, P0, t, h)
    print "K\n", K
    print "P\n", P