import numpy as np
import networkx as nx

from structures import *


a = [[ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
     [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]];
a = np.array(a)
a = nx.from_numpy_matrix(a)

s=[]
s.append(Clique(a,12))
s.append(Star(a,12))
s.append(BipartiteCore(a,12))
s.append(NearBipartiteCore(a,12))

for ss in s:
    ss.compute_mdl_cost()
    print ss.mdl_cost
