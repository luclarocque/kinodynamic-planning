from save_obj import load_object
from kripke import kripke
from modelchecker import modelcheck

# MG has 2 regions, each of their best paths has an endnode in the other region
MG = load_object('MG.pkl')
MG.graphs = load_object('MGgraphs.pkl')
endnodes = load_object('endnodes.pkl')
Props = ['p', 'q']

K = kripke(MG, endnodes, Props)

print modelcheck(K, 'nuZ.((q && muX.((p && Z) || <>X)) || (p && muY.((q && Z) || <>Y)))',
                 Props, MG.graphs[0].V[0][0])