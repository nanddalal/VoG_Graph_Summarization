import sys
import csv
import traceback
import numpy as np
import networkx as nx
import multiprocessing as mp
from operator import itemgetter

import structures


class VoG:
    def __init__(self, input_file):
        self.top_k_structures = []
        self.parse_adj_list_file(input_file)
        self.mdl_workers = mp.Pool(processes=(mp.cpu_count() * 2))
        self.perform_slash_burn(1)
        print self.top_k_structures

    # TODO: this assumes a 1 indexed adjacency list
    def parse_adj_list_file(self, input_file):
        print "Parsing adjacency list"

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
        self.total_num_edges = self.G.number_of_edges()

    def perform_slash_burn(self, k, gcc_num_nodes_criterion=1000):
        """ Peforms SlashBurn algorithm for subgraph generation
        
        Args:
            k: number of hubsets to remove at each iteration
            gcc_num_nodes_criterion: the inclusive upper-bound criterion for a subgraph to be a GCC which will be burned
        """

        print "Entering main Slash Burn loop"

        gcc_queue = [self.G]
        self.gamma = np.array([])  # deque

        while len(gcc_queue) > 0:
            current_gcc = gcc_queue[0]
            del gcc_queue[0]

            # 1
            # get a sorted list of (node, degree) in decreasing order
            k_hubset_nd = sorted(current_gcc.degree_iter(), key=itemgetter(1), reverse=True)
            # get the node index for the k highest degree vertex
            k_hubset = [i[0] for i in k_hubset_nd[0:k]]

            # consider subgraphs (stars) consisting of centrality nodes from the k hubset we are about to slash
            for node in k_hubset:
                hubset_subgraph = current_gcc.neighbors(node)
                hubset_subgraph.append(node)
                self.mdl_workers.apply_async(mdl_encoding,
                                             args=(current_gcc.subgraph(hubset_subgraph), self.total_num_nodes),
                                             callback=self.collect_mdl_results)

            # remove the k hubset from G, so now we have G' (slash!)
            current_gcc.remove_nodes_from(k_hubset)
            # add removed k hubset to the front of gamma
            self.gamma = np.insert(self.gamma, 0, k_hubset)

            # 2
            # get all the subgraphs after removing the k hubset
            sorted_sub_graphs = [(sub_graph, sub_graph.number_of_nodes())
                                 for sub_graph in nx.connected_component_subgraphs(current_gcc)]
            # TODO: making a copy - shouldn't make a copy
            # sort the subgraphs by the number of nodes in decreasing order
            sorted_sub_graphs = sorted(sorted_sub_graphs, key=itemgetter(1), reverse=True)

            # iterate over the remaining subgraphs we are "burning"
            for sub_graph, num_nodes in sorted_sub_graphs:
                if sub_graph.number_of_nodes() <= gcc_num_nodes_criterion:
                    self.mdl_workers.apply_async(mdl_encoding,
                                                 args=(sub_graph, self.total_num_nodes),
                                                 callback=self.collect_mdl_results)
                else:
                    # append the subgraph to GCCs queue
                    gcc_queue.append(sub_graph)
                # add the nodes in the non-GCC to the back of gamma
                self.gamma = np.append(self.gamma, sub_graph.nodes())

        self.mdl_workers.close()
        self.mdl_workers.join()

    def collect_mdl_results(self, result):
        self.top_k_structures.append(result)


def mdl_encoding(sub_graph, total_num_nodes):
    try:
        structure_types = [
            structures.Star(sub_graph, total_num_nodes),
            structures.BipartiteCore(sub_graph, total_num_nodes),
            structures.Error(sub_graph, total_num_nodes)
        ]
        for st in structure_types:
            st.compute_mdl_cost()
        optimal_structure = min(structure_types, key=lambda k: k.mdl_cost)
        return optimal_structure.mdl_cost
    except:
        # Put all exception text into an exception and raise that
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


if __name__ == '__main__':
    vog = VoG('./test_bipartite_core.txt')

