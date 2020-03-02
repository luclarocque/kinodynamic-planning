import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


# returns a list of lists of vertices of a rectangular prism starting with one
#  vertex at pos and with dimensions width, depth, height.
#  with array indices given as in the diagram below:

#       3________7
#       /.      /|
#      / .     / |
#    1/_______/5 |
#     |  .    |  |
#     | .2. . | ./6
#     |.      | /
#    0|_______|/4

def vertices(pos, width, depth, height):
    a = np.array(pos)
    lst = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                newv = tuple(a + np.array([width*i, depth*j, height*k]))
                lst.append(newv)
    return lst


def faces(pos, width, depth, height):
    v = vertices(pos, width, depth, height)
    faces = [[v[0], v[1], v[3], v[2]],
             [v[4], v[5], v[7], v[6]],
             [v[0], v[1], v[5], v[4]],
             [v[2], v[3], v[7], v[6]],
             [v[1], v[3], v[7], v[5]],
             [v[0], v[2], v[6], v[4]]]
    return faces


# plots a 3d rectangular prism on ax
def draw_3d(pos, width, depth, height, ax, color=(0, 0.7, 0.8, 0.15)):
    f = faces(pos, width, depth, height)
    ax.add_collection3d(Poly3DCollection(f, 
        facecolors=color, linewidths=0.5, edgecolors='k', alpha=.25))


# ensures trajectories stay within given list, intervals, of form (a,b)
def within_boundary(x, intervals):
    for i in range(3):
        try:
            pos = x.state[i]  # x is a Node
        except:
            pos = x[i]  # x is a tuple/list
        a = intervals[i][0]
        b = intervals[i][1]
        if not(a <= pos <= b):
            return False
    return True


# goal_interval produces a list of intervals, (a,b)
# goal is of the form:
# 3d object (rectangular prism)
#   [corner position (triple), width, depth, height]
def goal_intervals(goal):
    corner = goal[0]
    intervals = []
    for i in range(3):
        intervals.append((corner[i], corner[i]+goal[i+1]))
    return intervals

# def test():
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     ax.set_xlabel('X')
#     ax.set_xlim3d(0, 1)
#     ax.set_ylabel('Y')
#     ax.set_ylim3d(0, 1)
#     ax.set_zlabel('Z')
#     ax.set_zlim3d(0, 1)

#     draw_goal((0,0,0),1,1,1, ax)
#     ax.scatter3D([0.25, 0.5, 0.75], [0.1, 0.2, 0.4], [0.1, 0.3, 0.9])
#     plt.show()

if __name__ == "__main__":
    test()