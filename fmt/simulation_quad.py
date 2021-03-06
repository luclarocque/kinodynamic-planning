import numpy as np
from numpy import sin, cos, tan
from classes import Node
from fmt_quad import planner
from lqr import full_feedback
import matplotlib.pyplot as plt
from scipy import integrate
from numpy.linalg import multi_dot
from numpy.polynomial.polynomial import polyval, polyder

h = 0.005  # integration time step
dim = 12  # dimension of system
udim = 4  # dimension of control input

# constants
m = 0.3
g = 9.8
Ix = 0.00002
Iy = 0.00002
Iz = 0.00002
J = np.diag([Ix, Iy, Iz])

# ***controller constants***
# --very good--
# k_x = np.eye(3)*12/1.
# k_v = np.eye(3)*7/1.
# k_R = np.eye(3)*1/10.
# k_Om = np.eye(3)*1/500.
# -------------
# k_x = np.eye(3)*2/10.
# k_v = np.eye(3)*3/10.
# k_R = np.eye(3)*1./14
# k_R[0,0] = 1./5
# k_Om = np.eye(3)*1./595
#--------------
k_x = np.eye(3)*10/1.
k_v = np.eye(3)*6/1.
k_R = np.eye(3)*1./12
k_Om = np.eye(3)*1./600


# LQR Setup
# -----------------------------------------------------------------------------
A = np.zeros([dim, dim])
A[0:3, 6:9] = np.eye(3)
A[3:6, 9:12] = np.fliplr(np.eye(3))
A[6:8, 4:6] = np.array([[g, 0], [0, -g]])

B = np.zeros([dim, udim])
B[9:12, 1:4] = np.array([[1./Ix, 0, 0], [0, 1./Iy, 0], [0, 0, 1./Iz]])
B[8, 0] = 1./m

Q = np.eye(dim)*20.  # state cost matrix Q
# Q[0:3, 0:3] = np.eye(3)*1.  # position weight
# Q[3:6, 3:6] = np.eye(3)*1.  # angle weight
Q[6:9, 6:9] = np.eye(3)*25.  # translational velocity weight
# Q[9:12, 9:12] = np.eye(3)*1. # rotational velocity weight

R_lqr = np.eye(udim)*1e12  # control cost matrix R
R_lqr[0, 0] = 1.
# -----------------------------------------------------------------------------


# unit vector
e_3 = np.array([[0, 0, 1.]]).T


def S(x):
    return sin(x)


def C(x):
    return cos(x)


#  Rotation matrix in 3D, with angles: ps (yaw), th (pitch), ph (roll)
#  Determined by multiplying rotation matrices in the order Z,Y,X
def Rot(ps, th, ph):
    return np.array([
    [C(th)*C(ps), C(ps)*S(th)*S(ph)-S(ps)*C(ph), C(ps)*S(th)*C(ph)+S(ps)*S(ph)],
    [C(th)*S(ps), S(ps)*S(th)*S(ph)+C(ps)*C(ph), S(ps)*S(th)*C(ph)-C(ps)*S(ph)],
    [-S(th), S(ph)*C(th), C(ph)*C(th)]])


# the hat map takes a vector x and maps it to a skew symmetric matrix
#   such that:  np.dot(hat(x), y) = np.cross(x,y) for all x,y in R3
def hat(x):
    z = x[2]
    y = x[1]
    x = x[0]
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


# the vee map takes a skew symmetric matrix X (2D-arr) and maps it to a vector.
#   (Inverse of hat map)
def vee(X):
    y = X[0, 2]
    z = X[1, 0]
    x = X[2, 1]
    return np.array([x, y, z])


# maps real-valued angle a to the interval [-pi, pi]
def fix_angle(a):
    a = np.mod(a, 2*np.pi)
    if a > np.pi:
        a -= 2*np.pi
    return a


# x is a state, rescales angles using fix_angle function
def fix_state(x):
    x = list(x)
    a = [fix_angle(xi) for xi in x[3:6]]
    x[3:6] = np.array(a)
    return x


# plot actual vs desired traj.
#   ts: list of times
#   xpoly: list of polynomial coefficients for entire traj
#   xactual: list of state values for a particular component of state
#   T: list of time durations for each segment
#   ax: plot axes on which to plot
def plotter(ts, xpoly, xactual, T, ax):
    M = len(T)
    N = int(len(xpoly)/M - 1)
    tt = ts.copy()
    # print("ts\n", ts)
    for i in range(len(T)):
        curpoly = xpoly[i*(N+1):(i+1)*(N+1)]
        if i > 0: # don't want times to reset to 0 for each segment
            tt[i] = np.array(tt[i]) + tt[i-1][-1]  
        start = int(np.sum([len(ti) for ti in ts[:i]]))
        end = int(np.sum([len(ti) for ti in ts[:i+1]]))
        xref = list(polyval(ts[i], curpoly).reshape(len(ts[i]))) # use original ts
        ax.plot(tt[i], xref, color='g', linestyle='--')
        ax.plot(tt[i], xactual[start:end], color='k')


# plot norm of an error vector over time
def err_plotter(ts, errs, T, ax):
    ts = ts.copy()
    tlst = []
    errs = [np.linalg.norm(evec) for evec in errs]
    for i in range(len(T)):
        if i > 0: # don't want times to reset to 0 for each segment
            ts[i] = np.array(ts[i]) + ts[i-1][-1]
    tlst = [item for sublist in ts for item in sublist]
    ax.plot(tlst, np.zeros((len(errs),)), color='g')
    ax.plot(tlst, errs, color='k')


# -------- Dynamics --------

"""
state x = [[x, y, z, psi, theta, phi, xdot, ydot, zdot, p, q, r]].T
  where:
    x, y, z          -- position (World frame)
    psi, theta, phi  -- yaw, roll, pitch  (World frame)
    xdot, ydot, zdot -- translational velocity (World frame)
    p, q, r          -- angular velocities (Body frame)

    Note that we approximate [phidot, thetadot, psidot] == [p, q, r]


"""


def fcn(t, x, u):
    g17 = -1./m * ( cos(x[5])*cos(x[3])*sin(x[4]) + sin(x[5])*sin(x[3]) )
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
    res =  np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]) \
           + np.dot(G, u)
    return res
# --------------------------------------------------------------------------- #

#  x: current (initial) node
#  P: dictionary of 4 polynom trajectories (M arrays [p0 ... pN] concatenated)
#  T: list of segment times
#  ax: plot axes returned from fmt_quad planner
def simulate(x, P, T, ax):
    print("Simulating trajectory tracking...")
    px = P['x']
    py = P['y']
    pz = P['z']
    pyaw = P['yaw']

    #=======================================
    # The following are for testing tracking
    #=======================================
    # Simple trajectory examples (including steps and linear ramps) 
    T = [0.5, 4., 3., 3.]
    px = np.array([[1.5,0,0,0,0,0,0,0,0,0,
                    2.5,0,0,0,0,0,0,0,0,0,
                    2.5,-1.5/T[2],0,0,0,0,0,0,0,0,
                    1,0,0.2,0,0,0,0,0,0,0]]).T
    py = np.array([[1,0,0,0,0,0,0,0,0,0,
                    2,0,0,0,0,0,0,0,0,0,
                    2,-1./T[2],0,0,0,0,0,0,0,0,
                    1,0,0.2,0,0,0,0,0,0,0]]).T
    pz = np.array([[0,0,0,0,0,0,0,0,0,0,
                    1,0,0,0,0,0,0,0,0,0,
                    1,-1./T[2],0,0,0,0,0,0,0,0,
                    0,0,0.2,0,0,0,0,0,0,0]]).T
    pyaw = np.array([[0,0,0,0,0,0,0,0,0,0,
                      0,1./T[1],0,0,0,0,0,0,0,0,
                      1,0,0,0,0,0,0,0,0,0,
                      1,0,-0.1,0,0,0,0,0,0,0]]).T

    # T = [0.5, 1., 1.]
    # px = np.array([[1.5,0,0,0,0,0,0,0,0,0,
    #                 2.5,0,0,0,0,0,0,0,0,0,
    #                 2.5,0,0,0,0,0,0,0,0,0]]).T
    # py = np.array([[1,0,0,0,0,0,0,0,0,0,
    #                 2,0,0,0,0,0,0,0,0,0,
    #                 2,0,0,0,0,0,0,0,0,0]]).T
    # pz = np.array([[0,0,0,0,0,0,0,0,0,0,
    #                 1,0,0,0,0,0,0,0,0,0,
    #                 1,0,0,0,0,0,0,0,0,0]]).T
    # pyaw = np.array([[0,0,0,0,0,0,0,0,0,0,
    #                   0,0,0,0,0,0,0,0,0,0,
    #                   0,0,0,0,0,0,0,0,0,0]]).T
    #======================================

    M = len(T)
    N = int(len(px)/M - 1)
    P0 = np.eye(12)  # P0 used to integrate ricatti eqn (LQR only)
    t0 = 0.  # initial time

    # x.state = fix_state(x.state)

    newtraj = []
    ts = []
    Ex = []  # lists of errors
    Ev = []
    ER = []
    EOm = []
    ts.append([t0])
    for i in range(M):
        ode = integrate.ode(fcn)
        ode.set_initial_value(x.state, t0)
        ode.set_integrator('vode', nsteps=3000, method='bdf')
        newtraj.append([x.state])

        # extract polynomial coeffs for current traj segment
        x_d = px[i*(N+1):(i+1)*(N+1)]  # position
        y_d = py[i*(N+1):(i+1)*(N+1)]
        z_d = pz[i*(N+1):(i+1)*(N+1)]
        yaw_d = pyaw[i*(N+1):(i+1)*(N+1)]  # yaw
        yawdot_d = polyder(yaw_d)  # derivative of yaw
        yawddot_d = polyder(yawdot_d)  # second derivative of yaw
        vx_d = polyder(x_d)   # velocity
        vy_d = polyder(y_d)
        vz_d = polyder(z_d)
        ax_d = polyder(vx_d)  # acceleration
        ay_d = polyder(vy_d)
        az_d = polyder(vz_d)
        jx_d = polyder(ax_d)  # jerk
        jy_d = polyder(ay_d)
        jz_d = polyder(az_d)
        sx_d = polyder(jx_d)  # snap
        sy_d = polyder(jy_d)
        sz_d = polyder(jz_d)
        while ode.successful() and ode.t < T[i]:
            t = ode.t

            # position and velocity tracking errors:
            e_x = np.zeros((3, 1))
            e_x[0] =  polyval(t, x_d) - x.state[0]
            e_x[1] =  polyval(t, y_d) - x.state[1]
            e_x[2] =  polyval(t, z_d) - x.state[2]

            e_v = np.zeros((3, 1))
            e_v[0] =  polyval(t, vx_d) - x.state[6]
            e_v[1] =  polyval(t, vy_d) - x.state[7]
            e_v[2] =  polyval(t, vz_d) - x.state[8]

            # define derivatives of differentially flat output variables
            a_d = np.array([polyval(t, ax_d),  # acceleration
                            polyval(t, ay_d),
                            polyval(t, az_d)])
            j_d = np.array([polyval(t, jx_d),  # jerk
                            polyval(t, jy_d),
                            polyval(t, jz_d)])
            s_d = np.array([polyval(t, sx_d),  # snap
                            polyval(t, sy_d),
                            polyval(t, sz_d)])
            psi_d = polyval(t, yaw_d)
            psidot_d = polyval(t, yawdot_d)
            psiddot_d = polyval(t, yawddot_d)

            R = Rot(*x.state[3:6])  # actual rotation matrix: B to W
            b_3 = np.dot(R, e_3)  # actual third body-fixed axis
            Fn = (m*a_d + m*g*e_3)
            Fn_err = (m*a_d + m*g*e_3) + np.dot(k_x, e_x) + np.dot(k_v, e_v)
            u1ff = np.dot(b_3.T, Fn)

            # b2c = np.array([[-sin(psi_d), cos(psi_d), 0.]]).T
            # b3d = -Fn_err/np.linalg.norm(Fn_err)
            # b1d = (np.cross(b2c, b3d, axis=0) /
            #        np.linalg.norm(np.cross(b2c, b3d, axis=0)))
            # b2d = np.cross(b3d, b1d, axis=0)
            b1c = np.array([[cos(psi_d), sin(psi_d), 0.]]).T
            b3d = Fn_err/np.linalg.norm(Fn_err)
            b2d = (np.cross(b3d, b1c, axis=0) /
                   np.linalg.norm(np.cross(b3d, b1c, axis=0)))
            b1d = np.cross(b2d, b3d, axis=0)
            Rd = np.concatenate([b1d, b2d, b3d], axis=1)
            # print("Check Rd in SO(3): Rd.T Rd = {}".format(np.dot(Rd.T, Rd)))

            Om_b = np.asarray(x.state[9:12]).reshape(3,1)

            # to define Om_d (desired angular velocity)
            h_Om = m/u1ff * (np.dot(j_d.T, b3d)*b3d - j_d)
            pd = -np.dot(h_Om.T, b2d)
            qd = np.dot(h_Om.T, b1d)
            rd = psidot_d*np.dot(e_3.T, b3d)
            Om_d = pd*b1d + qd*b2d + rd*b3d

            # defining Omdot_d (desired angular acceleration)
            u1ffdot = -m*np.dot(j_d.T, b3d)
            u1ffddot = -np.dot(b3d.T, m*s_d + np.cross(Om_d, np.cross(Om_d, b3d, axis=0), axis=0))
            h_a = -1/u1ff * (m*s_d + u1ffddot*b3d 
                             + 2*u1ffdot*np.cross(Om_d, b3d, axis=0)
                             + np.cross(Om_d, np.cross(Om_d, b3d, axis=0), axis=0))
            a1d = -np.dot(h_a.T, b2d)
            a2d = np.dot(h_a.T, b1d)
            a3d = np.dot(e_3.T,psiddot_d*b3d - psidot_d*h_Om)

            Omdot_d = a1d*b1d + a2d*b2d + a3d*b3d

            # attitude tracking errors
            e_R = 1/2 * vee(np.dot(R.T, Rd) - np.dot(Rd.T, R))
            # e_Om = multi_dot([R.T, Rd, Om_d]) - Om_b
            e_Om = Om_d - Om_b


            # Proportional Controller
            # --------------------------------------------------------
            uvec_expr = (multi_dot([R.T, Rd, Omdot_d]) -
                         multi_dot([hat(Om_b), R.T, Rd, Om_d]))

            u = np.zeros((4, 1))
            u[0] = -np.dot(b_3.T, Fn_err)
            # u[1:4] = np.dot(J, uvec_expr) \
            #          + np.cross(Om_b, np.dot(J, Om_b), axis=0) \
            #          + np.dot(k_R, e_R) \
            #          + np.dot(k_Om, e_Om)
            u[1:4] = np.dot(k_R, e_R) + np.dot(k_Om, e_Om)
            # --------------------------------------------------------



            # LQR Controller
            # --------------------------------------------------------
            # K, P0, eigs = full_feedback(A, B, Q, R_lqr, P0, ode.t, h)
            # print("K\n", K)
            # print("eigs A-BK")
            # err = np.concatenate((e_x, e_v, e_R, e_Om))
            # u =  np.array(np.dot(-K, err))

            # u = np.zeros((4))
            # u[0] = -3.

            # print("Orig control: {}\nLQR adjust: {}\nResulting control: {}".format(e.control, LQR_term, u))
            # --------------------------------------------------------
            
            # print("_____"*5)
            # print("position:\n", )
            # print("||u||: ", np.linalg.norm(u))
            # print("e_x\n", e_x)
            # print("e_v\n", e_v)
            # print("e_R\n", e_R)
            # print("e_Om\n", e_Om)

            u = u.reshape(4)
            ode.set_f_params(u)
            Ex.append(e_x)
            Ev.append(e_v)
            ER.append(e_R)
            EOm.append(e_Om)
            ode.integrate(ode.t + h)
            ts[i].append(ode.t)
            new_state = ode.y  # fix_state(ode.y) 
            newtraj[i].append(new_state)
            x = Node(new_state)
        ts.append([])
    Ex.append(e_x)
    Ev.append(e_v)
    ER.append(e_R)
    EOm.append(e_Om)

# *** Visualization ***

    xs = []
    ys = []
    zs = []
    yaws = []
    # newtraj is a list of size M of lists of states (one list per segment), 
    #   i.e., newtraj = [[s01,s02,...], [s11,s12,...], ...]
    for lst in newtraj:
        for x in lst:
            xs.append(x[0])
            ys.append(x[1])
            zs.append(x[2])
            yaws.append(x[3])

    varfont = 16
    # plot norm of errors
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(221)
    ax1.set_xlabel("t", fontsize=varfont)
    ax1.set_ylabel("$\Vert  e_x \Vert$", fontsize=varfont)
    ax2 = fig2.add_subplot(222)
    ax2.set_xlabel("t", fontsize=varfont)
    ax2.set_ylabel("$\Vert e_v \Vert$", fontsize=varfont)
    ax3 = fig2.add_subplot(223)
    ax3.set_xlabel("t", fontsize=varfont)
    ax3.set_ylabel("$\Vert e_R \Vert$", fontsize=varfont)
    ax4 = fig2.add_subplot(224)
    ax4.set_xlabel("t", fontsize=varfont)
    ax4.set_ylabel("$\Vert e_{\Omega} \Vert$", fontsize=varfont)
    err_plotter(ts, Ex, T, ax1)
    err_plotter(ts, Ev, T, ax2)
    err_plotter(ts, ER, T, ax3)
    err_plotter(ts, EOm, T, ax4)

    # plot x,y,z,yaw vs reference
    fig1 = plt.figure()
    # THE FOLLOWING INCLUDES YAW
    # ax1 = fig1.add_subplot(221)
    # ax2 = fig1.add_subplot(222)
    # ax3 = fig1.add_subplot(223)
    # ax4 = fig1.add_subplot(224)
    # ax1.set_xlabel("t", fontsize=varfont)
    # ax1.set_ylabel("x", fontsize=varfont)
    # ax2.set_xlabel("t", fontsize=varfont)
    # ax2.set_ylabel("y", fontsize=varfont)
    # ax3.set_xlabel("t", fontsize=varfont)
    # ax3.set_ylabel("z", fontsize=varfont)
    # ax4.set_xlabel("t", fontsize=varfont)
    # ax4.set_ylabel("yaw", fontsize=varfont)
    # plotter(ts, px, xs, T, ax1)
    # plotter(ts, py, ys, T, ax2)
    # plotter(ts, pz, zs, T, ax3)
    # plotter(ts, pyaw, yaws, T, ax4)
    #--------------------------------
    # THE FOLLOWING OMITS YAW
    ax1 = fig1.add_subplot(311)
    ax2 = fig1.add_subplot(312)
    ax3 = fig1.add_subplot(313)
    ax1.set_xlabel("t", fontsize=varfont)
    ax1.set_ylabel("x", fontsize=varfont)
    ax2.set_xlabel("t", fontsize=varfont)
    ax2.set_ylabel("y", fontsize=varfont)
    ax3.set_xlabel("t", fontsize=varfont)
    ax3.set_ylabel("z", fontsize=varfont)
    plotter(ts, px, xs, T, ax1)
    plotter(ts, py, ys, T, ax2)
    plotter(ts, pz, zs, T, ax3)

    # plot actual trajectory in black
    ax.plot(xs, ys, zs, lw=3, color='k', zorder=1)
    plt.show()


if __name__ == '__main__':
    n = 200
    Jth = 260

    # xinit = Node((1., 1., 1., 0., 0., 0.))
    xinit = Node((1.5, 1., 0., 0., 0., 0.))
    Xgoal = [(6., 9., 8.), 1, 1, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_xlim3d(0, 10)
    ax.set_ylabel('Y')
    ax.set_ylim3d(0, 10)
    ax.set_zlabel('Z')
    ax.set_zlim3d(0, 10)

    P, T, z = planner(n, Jth, xinit, Xgoal, ax, plot_waypts=False)
    # recast xinit into 12D node
    xinit = Node([*xinit.state, 0., 0., 0., 0., 0., 0.])
    simulate(xinit, P, T, ax)