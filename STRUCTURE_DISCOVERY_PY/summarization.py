from operator import itemgetter
import csv
import numpy as np
import networkx as nx
import multiprocessing as mp
from math import log


class VoG:

    def __init__(self, input_file):
        self.mdl_results = []
        self.parse_adj_list_file(input_file)
        self.mdl_workers = mp.Pool(processes=(mp.cpu_count() * 2))
        self.slash_burn_multiple_gcc(1)

    # TODO: this assumes a 1 indexed adjacency list
    def parse_adj_list_file(self, input_file):
        with open(input_file) as gf:
            r = csv.reader(gf, delimiter=',')
            adj_list = np.array(list(r), int)

        adj_mat = np.zeros((adj_list.max(), adj_list.max()))
        adj_list -= 1  # TODO: fix this!!!
        for e in adj_list:
            adj_mat[e[0], e[1]] = 1
            adj_mat[e[1], e[0]] = 1

        self.G = nx.from_numpy_matrix(adj_mat)
        self.total_num_nodes = self.G.number_of_nodes()

    def slash_burn(self, k):
        self.gamma = np.array([])  # deque
        self.candidate_structures = []

        while self.G.number_of_nodes() > k:
            # 1
            # get a sorted list of (node, degree) in decreasing order
            k_hubset_nd = sorted(self.G.degree_iter(), key=itemgetter(1), reverse=True)[0:k]
            k_hubset = [i[0] for i in k_hubset_nd]  # get the node index for the k highest degree vertex
            self.G.remove_nodes_from(k_hubset)  # remove the k hubset from G, so now we have G' (slash!)
            self.gamma = np.insert(self.gamma, 0, k_hubset)  # add removed k hubset to the front of gamma

            # 2
            # get all the subgraphs after removing the k hubset
            sorted_sub_graphs = [(sub_graph, sub_graph.number_of_nodes())
                                 for sub_graph in nx.connected_component_subgraphs(self.G)]
            # sort the subgraphs by the number of nodes in decreasing order
            sorted_sub_graphs = sorted(sorted_sub_graphs, key=itemgetter(1), reverse=True)

            # because we have sorted, the zeroth subgraph is the greatest connected component
            GCC = sorted_sub_graphs[0][0]
            for sub_graph, num_nodes in sorted_sub_graphs[1:]:  # iterate over the remaining subgraphs we are "burning"
                self.candidate_structures.append(sub_graph)
                # add the nodes in the non-GCC to the back of gamma
                self.gamma = np.append(self.gamma, sub_graph.nodes())

            # 3
            self.G = GCC

    def slash_burn_multiple_gcc(self, k, gcc_num_nodes_criterion=1000):
        """ Peforms SlashBurn algorithm for subgraph generation
        
        Args:
            k: number of hubsets to remove at each iteration
            gcc_num_nodes_criterion: the inclusive upper-bound criterion for a subgraph to be a GCC which will be burned
        """

        GCCs = [self.G]  # all the GCCs that meet the criterion, acts as a queue
        self.gamma = np.array([])  # deque
        self.candidate_structures = []

        while len(GCCs) > 0:
            # TODO: O(n) operation, but we might still have to do BFS instead of DFS - global for external access?
            current_GCC = GCCs[0]  # extract the first element of GCCs
            GCCs = GCCs[1:]  # remove the first element of GCCs

            # 1
            # get a sorted list of (node, degree) in decreasing order
            k_hubset_nd = sorted(current_GCC.degree_iter(), key=itemgetter(1), reverse=True)[0:k]
            k_hubset = [i[0] for i in k_hubset_nd]  # get the node index for the k highest degree vertex

            # consider subgraphs (stars) consisting of centrality nodes from the k hubset we are about to slash
            for node in k_hubset:
                hubset_subgraph = current_GCC.neighbors(node)
                hubset_subgraph.append(node)
                # TODO: should we consider the size of these subgraphs or just add them to candidate structures?
                self.candidate_structures.append(current_GCC.subgraph(hubset_subgraph))
                mdl_encoding(self.candidate_structures[-1], self.total_num_nodes)
                # self.mdl_workers.apply_async(mdl_encoding,
                #                              args=(self.candidate_structures[-1], self.total_num_nodes),
                #                              callback=self.collect_mdl_results)

            current_GCC.remove_nodes_from(k_hubset)  # remove the k hubset from G, so now we have G' (slash!)
            self.gamma = np.insert(self.gamma, 0, k_hubset)  # add removed k hubset to the front of gamma

            # 2
            # get all the subgraphs after removing the k hubset
            sorted_sub_graphs = [(sub_graph, sub_graph.number_of_nodes())
                                 for sub_graph in nx.connected_component_subgraphs(current_GCC)]
            # TODO: making a copy - shouldn't make a copy
            # sort the subgraphs by the number of nodes in decreasing order
            sorted_sub_graphs = sorted(sorted_sub_graphs, key=itemgetter(1), reverse=True)

            for sub_graph, num_nodes in sorted_sub_graphs:  # iterate over the remaining subgraphs we are "burning"
                if sub_graph.number_of_nodes() <= gcc_num_nodes_criterion:
                    # self.mdl_workers.apply_async(mdl_encoding,
                    #                              args=(sub_graph, self.total_num_nodes),
                    #                              callback=self.collect_mdl_results)
                    mdl_encoding(sub_graph, self.total_num_nodes)
                    self.candidate_structures.append(sub_graph)  # meets the criterion, goes to candidate structures
                else:
                    GCCs.append(sub_graph)  # append the subgraph to GCCs queue
                # add the nodes in the non-GCC to the back of gamma
                self.gamma = np.append(self.gamma, sub_graph.nodes())

        self.mdl_workers.close()
        self.mdl_workers.join()

    def collect_mdl_results(self, result):
        print result
        self.mdl_results.append(result)


def mdl_encoding(sub_graph, total_num_nodes):
    mdl = MdlEncoding(sub_graph, total_num_nodes)
    return 5


class MdlEncoding:
    def __init__(self, graph, total_num_nodes):
        self.graph = graph
        self.total_num_nodes = total_num_nodes

        print "STAR", self.encode_star()
        print "BIPARTITE CORE", self.encode_bipartite_core()

    def Ln(self, n):
        c = log(2.865064, 2)
        logterm = log(n, 2)
        while logterm > 0:
            c = c + logterm
            logterm = log(logterm, 2)
        return c

    def nll(self, incl, excl, sub):
        if sub == 0:
            l = -log((excl / float(incl + excl)), 2)
        elif sub == 1:
            l = -log((incl / float(incl + excl)), 2)
        return l

    def lnu_opt(self, e_inc, e_exc):
        c_err = self.Ln(e_inc) \
                + e_inc*self.nll(e_inc, e_exc, 1) \
                + e_exc*self.nll(e_inc, e_exc, 0)
        return c_err
    
    def l2cnk(self, n, k):
        nbits = 0
        for i in range(n, n-k, -1):
            nbits = nbits + log(i, 2)
        
        for i in range(k, 0, -1):
            nbits = nbits - log(i, 2)
        return nbits

    # TODO: make this efficient
    def encode_star(self):
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
            mdl_cost = self.Ln(num_nodes-1) \
                       + log(self.total_num_nodes, 2) \
                       + self.l2cnk(self.total_num_nodes-1, num_nodes-1)
        else:
            mdl_cost = self.Ln(num_nodes-1) \
                       + log(self.total_num_nodes, 2) \
                       + self.l2cnk(self.total_num_nodes-1, num_nodes-1) \
                       + self.lnu_opt(E[0], E[1])  # TODO: not clear why we need this line

        return mdl_cost

    def encode_bipartite_core(self):
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
        MDLcost = self.Ln(k) \
                  + self.Ln(l) \
                  + self.l2cnk(N_tot, k) \
                  + self.l2cnk(N_tot - k, l)

        return MDLcost


if __name__ == '__main__':
    # vog = VoG('sb_paper_graph.txt')
    # vog = VoG('../DATA/cliqueStarClique.out')
    vog = VoG('./test_bipartite_core.txt')
