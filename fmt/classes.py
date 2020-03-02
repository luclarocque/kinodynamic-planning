class Node:
    state = (0., 0., 0., 0., 0., 0.)
    cost = 0.
    parent = None
    children = set()
    N_in = None
    N_out = None

    def __init__(self, statex):
        self.state = statex


class Edge:
    source = None
    target = None
    cost = 0.

    def __init__(self, srcnode, tarnode):
        self.source = srcnode
        self.target = tarnode