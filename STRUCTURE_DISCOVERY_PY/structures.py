from math import log
import numpy as np
import networkx as nx
from operator import itemgetter


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


def mdl_encoding(subgraph, total_num_nodes):
    err = Error(subgraph)
    err.compute_mdl_cost()
    structure_types = [
        Clique(subgraph, total_num_nodes),
        Star(subgraph, total_num_nodes),
        BipartiteCore(subgraph, total_num_nodes),
        NearBipartiteCore(subgraph, total_num_nodes),
        Chain(subgraph, total_num_nodes),
    ]
    for st in structure_types:
        st.compute_mdl_cost()
        st.benefit = err.mdl_cost - st.mdl_cost
    err.benefit = 0
    structure_types.append(err)
    optimal_structure = min(structure_types, key=lambda k: k.mdl_cost)
    return optimal_structure


class Clique:
    def __init__(self, graph, total_num_nodes):
        self.graph = graph
        self.total_num_nodes = total_num_nodes

    def __str__(self):
        return "fc " + ' '.join(map(str, sorted(map(int, self.nodes+1))))

    def compute_mdl_cost(self):
        Asmall = nx.to_scipy_sparse_matrix(self.graph)
        # count_nonzero = len(Asmall.nonzero()[0])
        count_nonzero = np.count_nonzero(Asmall)
        # E = (len(Asmall)**2 - len(Asmall) - np.count_nonzero(Asmall), np.count_nonzero(Asmall))
        len_Asmall = Asmall.shape[0]
        E = (len_Asmall ** 2 - len_Asmall - count_nonzero, count_nonzero)
        if E[0] == 0 or E[1] == 0:  # no excluded edges
            mdl_cost = ln(len_Asmall) + l2cnk(self.total_num_nodes, len_Asmall)
        else:
            mdl_cost = ln(len_Asmall) + l2cnk(self.total_num_nodes, len_Asmall) + lnu_opt(E[0], E[1])

        self.mdl_cost = mdl_cost

        self.nodes = np.array(self.graph.nodes())


class Star:
    def __init__(self, graph, total_num_nodes):
        self.graph = graph
        self.total_num_nodes = total_num_nodes

    def __str__(self):
        satellite_node_indexes = np.array(self.satellite_node_indexes)+1
        return "st " + str(self.max_degree_node_idx+1) + ", " + \
               ' '.join(map(str, sorted(map(int, satellite_node_indexes))))

    # TODO: make this efficient
    def compute_mdl_cost(self):
        # sorted list of node degree tuples in format (node, degree) in descending order
        star_node_degrees = sorted(self.graph.degree_iter(), key=itemgetter(1), reverse=True)
        num_nodes = len(star_node_degrees)

        if num_nodes <= 3:
            self.mdl_cost = np.inf
            return

        # remove star hub from node, degree list
        self.max_degree_node_idx = star_node_degrees[0][0]
        del star_node_degrees[0]

        num_missing_edges = (num_nodes - 1 - len(self.graph.neighbors(self.max_degree_node_idx)))
        self.satellite_node_indexes = [d[0] for d in star_node_degrees]
        num_extra_edges = self.graph.subgraph(self.satellite_node_indexes).number_of_edges()*2
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
                       + lnu_opt(E[0], E[1])

        self.mdl_cost = mdl_cost


class BipartiteCore:
    def __init__(self, graph, total_num_nodes):
        self.graph = graph
        self.total_num_nodes = total_num_nodes

    def __str__(self):
        return "bc " + ' '.join(map(str, self.left_side)) + ', ' + ' '.join(map(str, self.right_side))

    def compute_mdl_cost(self):
        Asmall = nx.to_numpy_matrix(self.graph)
        h = -0.01
        positive = 0.01
        negative = -0.01
        a = 4 * (h**2) / (1 - 4 * (h**2))
        c = 2 * h / (1 - 4 * (h**2))
        n = len(Asmall)
        if n < 3:
            self.mdl_cost = np.inf
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
        b = np.array(b).flatten()

        set1 = np.array(b > 0)
        set2 = np.array(b < 0)
        Einc = 2*(set1.sum()*set2.sum()-np.count_nonzero(Asmall[set1][:, set2])) + np.count_nonzero(Asmall[set1][:, set1]) + np.count_nonzero(Asmall[set2][:, set2])
        Eexc = len(Asmall)**2 - Einc
        # M = np.zeros((n, n))
        # M[set1][:, set2] = 1
        # M[set2][:, set1] = 1
        # E = np.logical_xor(M, Asmall)

        N_tot = self.total_num_nodes
        n_sub = set1.sum()
        E = [Einc, Eexc]
        n_sub2 = set2.sum()

        k = n_sub
        l = n_sub2
        if E[0] == 0 or E[1] == 0:
            mdl_cost = ln(k) \
                       + ln(l) \
                       + l2cnk(N_tot, k) \
                       + l2cnk(N_tot - k, l)
        else:
            mdl_cost = ln(k) \
                       + ln(l) \
                       + l2cnk(N_tot, k) \
                       + l2cnk(N_tot - k, l) \
                       + lnu_opt(E[0], E[1])

        self.mdl_cost = mdl_cost

        nodes = np.array(self.graph.nodes())
        self.left_side = list(nodes[np.nonzero(set1)] + 1)
        self.left_side.sort()
        self.right_side = list(nodes[np.nonzero(set2)] + 1)
        self.right_side.sort()


class NearBipartiteCore:
    def __init__(self, graph, total_num_nodes):
        self.graph = graph
        self.total_num_nodes = total_num_nodes

    def __str__(self):
        return "nb " + ' '.join(map(str, self.left_side)) + ', ' + ' '.join(map(str, self.right_side))

    def compute_mdl_cost(self):
        Asmall = nx.to_numpy_matrix(self.graph)
        h = -0.01
        positive = 0.01
        negative = -0.01
        a = 4 * (h**2) / (1 - 4 * (h**2))
        c = 2 * h / (1 - 4 * (h**2))
        n = len(Asmall)
        if n < 3:
            self.mdl_cost = np.inf
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
        b = np.array(b).flatten()
        set1 = np.array(b > 0)
        set2 = np.array(b < 0)
        Einc = np.count_nonzero(Asmall[set1][:, set1]) + np.count_nonzero(Asmall[set2][:, set2])
        Eexc = len(Asmall)**2 - Einc
        # M = np.zeros((n, n))
        # M[set1][:, set2] = 1
        # M[set2][:, set1] = 1
        # E = np.logical_xor(M, Asmall)

        N_tot = self.total_num_nodes
        n_sub = set1.sum()
        E = [Einc, Eexc]
        n_sub2 = set2.sum()

        k = n_sub
        l = n_sub2

        edges_inc = np.count_nonzero(Asmall[set1][:, set2])
        edges_exc = set1.sum()*set2.sum()-np.count_nonzero(Asmall[set1][:, set2])
        if E[0] == 0 or E[1] == 0:
            mdl_cost = ln(k) \
                       + ln(l) \
                       + l2cnk(N_tot, k) \
                       + l2cnk(N_tot - k, l)
        else:
            if edges_inc == 0 or edges_exc == 0:
                mdl_cost = ln(k) \
                           + ln(l) \
                           + l2cnk(N_tot, k) \
                           + l2cnk(N_tot - k, l) \
                           + lnu_opt(E[0], E[1])
            else:
                mdl_cost = ln(k) \
                           + ln(l) \
                           + l2cnk(N_tot, k) \
                           + l2cnk(N_tot - k, l) \
                           + np.log2((n_sub+n_sub2)**2) \
                           + edges_inc * nll(edges_inc, edges_exc, 1) \
                           + edges_exc * nll(edges_inc, edges_exc, 0) \
                           + lnu_opt(E[0], E[1])

        self.mdl_cost = mdl_cost

        nodes = np.array(self.graph.nodes())
        self.left_side = list(nodes[np.nonzero(set1)]+1)
        self.left_side.sort()
        self.right_side = list(nodes[np.nonzero(set2)]+1)
        self.right_side.sort()


class Chain:
    def __init__(self, graph, total_num_nodes):
        self.graph = graph
        self.total_num_nodes = total_num_nodes

    def __str__(self):
        return "ch " + ' '.join(map(str, sorted(map(int, self.nodes+1))))

    def compute_mdl_cost(self):
        if nx.number_of_nodes(self.graph) < 3:
            self.mdl_cost = np.inf  # set the mdl_cost as Chain to be maximum so it won't be chosen
            return
        Asmall = nx.to_numpy_matrix(self.graph)
        deg = np.sum(Asmall, axis=0)
        min_deg_ind = np.argmin(deg)
        p_init, empty_chain = self.bfs(Asmall, min_deg_ind, path=False)
        p_fin, chain = self.bfs(Asmall, p_init, path=True)
        missing = 0
        existing = 0
        for i in range(0, len(chain) - 1):
            if Asmall[chain[i], chain[i+1]] == 0:
                missing += 1
            else:
                existing += 1
        E_0 = 2 * missing + (np.count_nonzero(Asmall) - 2 * existing)
        E = (E_0, nx.number_of_nodes(self.graph) ** 2 - E_0)
        x = list(xrange(nx.number_of_nodes(self.graph)))
        n_tot_vec = self.total_num_nodes * np.ones((nx.number_of_nodes(self.graph),), dtype=np.int)
        if E[0] == 0 or E[1] == 0:
            mdl_cost = ln(nx.number_of_nodes(self.graph) - 1) + sum(np.log2(n_tot_vec - x))
        else:
            mdl_cost = ln(nx.number_of_nodes(self.graph) - 1) + sum(np.log2(n_tot_vec - x) ) + lnu_opt(E[0], E[1])

        self.mdl_cost = mdl_cost

        self.nodes = np.array(self.graph.nodes())

    def bfs(self, Asmall, start, path=True):
        queue = [start]
        chain = []
        extra_nodes_search = np.ones((nx.number_of_nodes(self.graph),), dtype=np.int)
        node_list = -1 * np.ones((nx.number_of_nodes(self.graph),), dtype=np.int)
        node_list[start] = start

        while queue:
            neighbors = np.array(np.nonzero(Asmall[queue[0], :])[1])  # get neighbors as numpy array
            for i in range(0, np.size(neighbors)):
                if node_list[neighbors[i]] == -1: # TODO: Die here
                    node_list[neighbors[i]] = queue[0]
                    queue.append(neighbors[i])
            qsize = len(queue)
            furthest_node = queue[qsize - 1]
            queue = queue[1:]
        if path:
            curr = furthest_node
            while curr != start:
                chain.append(curr)
                extra_nodes_search[curr] = 0
                curr = node_list[curr]
            chain.append(start)
            chain = list(reversed(chain))
            extra_nodes_search[start] = 0

        return furthest_node, chain


class Error:
    def __init__(self, graph):
        self.graph = graph

    def __str__(self):
        raise NotImplementedError

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

