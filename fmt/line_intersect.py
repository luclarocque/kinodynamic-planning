import numpy as np
from plot_3d import faces

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


# Return true if line segments AB and CD do NOT intersect
#   (where CD is any obstacle boundary line)
def collision_free(nodeA, nodeB, OBS):
    A=(nodeA.state[0],nodeA.state[1])
    B=(nodeB.state[0],nodeB.state[1])
    for o in OBS:
        obs=(o[0],o[1],o[0]+o[2],o[1]+o[3])
        C1=(obs[0],obs[1])
        D1=(obs[0],obs[3])
        C2=(obs[0],obs[1])
        D2=(obs[2],obs[1])
        C3=(obs[2],obs[3])
        D3=(obs[2],obs[1])
        C4=(obs[2],obs[3])
        D4=(obs[0],obs[3])
        inst1= ccw(A,C1,D1) != ccw(B,C1,D1) and ccw(A,B,C1) != ccw(A,B,D1) 
        inst2= ccw(A,C2,D2) != ccw(B,C2,D2) and ccw(A,B,C2) != ccw(A,B,D2)
        inst3= ccw(A,C3,D3) != ccw(B,C3,D3) and ccw(A,B,C3) != ccw(A,B,D3)
        inst4= ccw(A,C4,D4) != ccw(B,C4,D4) and ccw(A,B,C4) != ccw(A,B,D4)
        if not(inst1==False and inst2==False and inst3==False and inst4==False):
            return False
    return True

#  return a Bool indicating whether line segment from v1 to v2 intersects
#      rectangle with vertices a,b,c (ensure a and c are opposite,
#      across the diagonal). The 4th vertex is determined by the others.
def intersect(a, b, c, v0, v1, epsilon=1e-6):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    v0 = np.array(v0)
    v1 = np.array(v1)

    p_no = np.cross(b-a, c-b)
    u = v1 - v0
    dot = np.dot(p_no, u)

    if abs(dot) > epsilon:
        # if 'fac' is between [0 - 1] the point intersects with the segment.
        # otherwise:
        #  < 0.0: behind v0
        #  > 1.0: ahead of v1.
        w = v0 - a
        fac = -np.dot(p_no, w) / dot
        u = fac*u
        int_pt = v0 + u  # point of intersection
        #  if the dot product is <0 then int_pt is not in the rectangle
        int_in_rect = np.dot(int_pt-a, c-int_pt) >= epsilon and (0<=fac<=1)
        return int_in_rect
    else:
        # The segment is parallel to plane
        return False


# returns True if there are no collisions.
def collision_check_3d(n0, n1, OBS):
    try:
        # see if n0, n1 are nodes
        v0 = n0.state[:3]
        v1 = n1.state[:3]
    except:
        # otherwise, assume they are tuples/arrays
        v0 = n0
        v1 = n1
    for o in OBS:
        fs = faces(*o)
        for f in fs:
            if intersect(f[0], f[1], f[2], v0, v1) == True:
                return False
    return True


#  return a list of 6 faces given a 3d obstacle o
# def 3d_faces(o):

if __name__ == "__main__":
    pass
