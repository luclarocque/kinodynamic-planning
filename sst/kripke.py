import sst_generic_parallel as sst

# takes in a prop and the set of all atomic propositions (Props) and returns
#   a function which evaluates to true only for nodes in the region associated to prop
def prop_bool(prop, Props):
    parity = True
    if prop[0] == '!':
        parity = False
        prop = prop[1]
    for i in range(len(Props)):
        coords = sst.init_coord_list
        sizes = sst.regionsize_list
        if prop == Props[i]:
            x,y = coords[i][0:2]
            w,h = sizes[i]
            return lambda (node): parity == (x-w/2. < node.state[0] < x+w/2.) and \
                                            (y-h/2. < node.state[1] < y+h/2.)
    return lambda (node): False

def kripke(MG, endnodes, Props):
    K = sst.Graph([],[],[])
    # Initialize K with the initial nodes of each graph in MG

    # NOTE: We will use Graph.S to store the atomic proposition(s) associated with a given region (acts as L)
    for i in range(sst.num_props):
        xinit = sst.Node(MG.graphs[i].V[0][0].state)
        K.V.append(xinit)
        K.S.append(Props[i])
    # We now seek to add an edge between these initial nodes ONLY if the appropriate 
    #   endnode is found in its desired goal region
    endnode_idx = 0
    for i in range(sst.num_props):
        for j in range(sst.num_props):
            if i == j: continue;
            else:
                cur_prop = Props[j]
                # does endnode at endnode_idx satisfy the prop of the associated goalregion
                L = prop_bool(cur_prop, Props)
                if L(endnodes[endnode_idx]):  # TODO: 
                    K.V[i].children = K.V[i].children + [K.V[j]]
                    new_edge = sst.Edge([], K.V[i], K.V[j], 0, 0, 0)
                    K.E.append(new_edge)
                endnode_idx += 1
    return K


# Edge(state_traj, srcnode, tarnode, controlval, t, trajcost):