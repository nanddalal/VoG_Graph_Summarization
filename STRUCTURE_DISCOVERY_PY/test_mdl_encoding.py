import numpy as np
import networkx as nx

from structures import *


a = [[ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
     [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],
     [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],
     [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],
     [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.],
     [ 1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.],
     [ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.],
     [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.]];
cs = 50
a = np.ones((cs,cs)) - np.eye(cs)
a = np.array(a)
a = nx.from_numpy_matrix(a)

tnn = 12000

s=[]
s.append(Clique(a,tnn))
s.append(Star(a,tnn))
s.append(BipartiteCore(a,tnn))
s.append(NearBipartiteCore(a,tnn))
s.append(Error(a))

for ss in s:
    ss.compute_mdl_cost()
    print ss.mdl_cost
