from save_obj import load_object
from kripke import kripke
from modelchecker import modelcheck

# MG has 2 regions, each of their best paths has an endnode in the other region
MG = load_object('MG.pkl')
MG.graphs = load_object('MGdoubleint2.pkl')
endnodes = load_object('endnodes_doubleint2.pkl')
Props = ['a', 'b', 'c']

K = kripke(MG, endnodes, Props)

print modelcheck(K, 'nuZ.((a && muX.(((b || c) && Z) || <>X)) || (b && muY.(((a || c) && Z) || <>Y)) || (c && muY.(((a || b) && Z) || <>Y)))',
                 Props, MG.graphs[0].V[0][0])

#nuZ.((a && muX.(((b || c) && Z) || <>X)) || (b && muY.(((a || c) && Z) || <>Y)) || (c && muY.(((a || b) && Z) || <>Y)))