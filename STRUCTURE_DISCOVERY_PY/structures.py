from operator import itemgetter
from math import log
import numpy as np
import networkx as nx


def ln(n):
    c = log(2.865064, 2)
    logterm = log(n, 2)
    while logterm > 0:
        c += logterm
        logterm = log(logterm, 2)
    return c


def nll(incl, excl, sub):
    if sub == 0:
        l = -log((excl / float(incl + excl)), 2)
    elif sub == 1:
        l = -log((incl / float(incl + excl)), 2)
    return l


def lnu_opt(e_inc, e_exc):
    c_err = ln(e_inc) \
            + e_inc*nll(e_inc, e_exc, 1) \
            + e_exc*nll(e_inc, e_exc, 0)
    return c_err


def l2cnk(n, k):
    nbits = 0
    for i in range(n, n-k, -1):
        nbits += log(i, 2)

    for i in range(k, 0, -1):
        nbits -= log(i, 2)
    return nbits


class Star:
    def __init__(self, graph, total_num_nodes):
        self.graph = graph
        self.total_num_nodes = total_num_nodes

    # TODO: make this efficient
    def compute_mdl_cost(self):
        # sorted list of node degree tuples in format (node, degree) in descending order
        star_node_degrees = sorted(self.graph.degree_iter(), key=itemgetter(1), reverse=True)
        num_nodes = len(star_node_degrees)

        if num_nodes <= 3:
            return

        # remove star hub from node, degree list
        max_degree_node_idx = star_node_degrees[0][0]
        del star_node_degrees[0]

        num_missing_edges = (num_nodes - 1 - len(self.graph.neighbors(max_degree_node_idx)))
        satellite_node_indexes = [d[0] for d in star_node_degrees]
        num_extra_edges = self.graph.subgraph(satellite_node_indexes).number_of_edges()*2
        num_non_star_edges = 2*num_missing_edges + num_extra_edges
        E = (num_non_star_edges, (num_nodes**2 - num_non_star_edges))

        if E[0] == 0 or E[1] == 0:
            mdl_cost = ln(num_nodes-1) \
                       + log(self.total_num_nodes, 2) \
                       + l2cnk(self.total_num_nodes-1, num_nodes-1)
        else:
            mdl_cost = ln(num_nodes-1) \
                       + log(self.total_num_nodes, 2) \
                       + l2cnk(self.total_num_nodes-1, num_nodes-1) \
                       + lnu_opt(E[0], E[1])  # TODO: not clear why we need this line

        self.mdl_cost = mdl_cost


class BipartiteCore:
    def __init__(self, graph, total_num_nodes):
        self.graph = graph
        self.total_num_nodes = total_num_nodes

    def compute_mdl_cost(self):
        Asmall = nx.to_numpy_matrix(self.graph)
        h = -0.01
        positive = 0.01
        negative = -0.01
        a = 4 * (h**2) / (1 - 4 * (h**2))
        c = 2 * h / (1 - 4 * (h**2))
        n = len(Asmall)
        if n < 3:
            return
        deg = np.array(Asmall.sum(axis=0)).flatten()
        D = np.diag(deg)
        matI = np.eye(n)
        phi = np.zeros(n)
        idx = np.argmax(deg)
        neighbors = np.array(Asmall[idx]).flatten().nonzero()
        phi[idx] = positive
        phi[neighbors] = negative
        b = np.dot(np.linalg.inv(matI + a * D - c * Asmall), phi)

        set1 = np.array((b > 0)).flatten()
        set2 = np.array((b < 0)).flatten()
        M = np.zeros((n, n))
        # TODO: cannot seem to find a way to do logical slicing like in Matlab
        for i in range(n):
            for j in range(n):
                if set1[i] and set2[j]:
                    M[i, j] = 1
        E = np.logical_xor(M, Asmall)

        N_tot = self.total_num_nodes
        n_sub = sum(set1)
        E = np.array(E)
        n_sub2 = sum(set2)

        k = n_sub
        l = n_sub2
        # TODO: how do we run lnu_opt - look at near bipartite core code
        mdl_cost = ln(k) \
                   + ln(l) \
                   + l2cnk(N_tot, k) \
                   + l2cnk(N_tot - k, l)

        self.mdl_cost = mdl_cost


class Error:
    def __init__(self, graph, total_num_nodes):
        self.graph = graph
        self.total_num_nodes = total_num_nodes

    def compute_mdl_cost(self):
        Asmall = nx.to_numpy_matrix(self.graph)
        E = (np.count_nonzero(Asmall), len(Asmall)**2 - np.count_nonzero(Asmall))
        if E[0] != 0 and E[1] != 0:
            mdl_cost = lnu_opt(E[0], E[1])
        elif E[0] != 0:
            mdl_cost = ln(E[0])
        elif E[1] != 0:
            mdl_cost = ln(E[1])
        self.mdl_cost = mdl_cost

