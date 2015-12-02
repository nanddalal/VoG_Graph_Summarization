#!/usr/bin/env python
import threading
import os
import time
import signal
import sys
import traceback
import csv
import math
import heapq
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
import multiprocessing as mp
# import matplotlib.pyplot as plt
from operator import itemgetter

import structures
import cProfile

lock = threading.Lock()

def profiler(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func


class VoGTimeout(Exception):
    @staticmethod
    def time_limit_handler(signum, frame):
        print "Reached specified time limit"
        raise VoGTimeout


class VoG:
    def __init__(self, input_file, slash_burn_k=1, top_k=10, time_limit=None, parallel=True):
        self.top_k = top_k
        self.top_k_structures = []

        print "Parsing adjacency list"
        self.parse_adj_list_file(input_file)
        # self.visualize_graph()

        self.parallel = parallel
        if parallel:
            self.workers = mp.Pool(processes=(mp.cpu_count() * 2))

        if time_limit is not None:
             signal.signal(signal.SIGALRM, VoGTimeout.time_limit_handler)
             signal.alarm(time_limit)

        try:
            print "Performing slash burn"
            # self.perform_slash_burn(slash_burn_k, int(math.log(self.total_num_nodes)))
            self.perform_slash_burn(slash_burn_k, 7)
        except VoGTimeout:
            pass  # TODO: probably need to be doing something here
        else:
            signal.alarm(0)  # TODO: understand why this is necessary (it may not be)

        print "Printing top k structures"
        self.print_top_k_structures()
        # self.visualize_graph()
        # plt.show()
        # plt.close()

    def visualize_graph(self):
        # fig = plt.figure()
        pos = nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos)
        nx.draw_networkx_edges(self.G, pos)
        nx.draw_networkx_labels(self.G, pos)

    def print_top_k_structures(self):
        self.top_k_structures.sort(reverse=True)
        for s in self.top_k_structures:
            print s[1].__class__.__name__, s[1].graph.nodes()

    # TODO: this assumes a 1 indexed adjacency list
    def parse_adj_list_file(self, input_file):
        with open(input_file) as gf:
            r = csv.reader(gf, delimiter='\t')
            adj_list = np.array(list(r), int)

        # adj_mat = np.zeros((adj_list.max(), adj_list.max()))
   	# adj_list -= 1 
        row, col, data = [], [], []
        for e in adj_list:
            row.append(e[0])
            col.append(e[1])
            data.append(1)

        adj_mat = csr_matrix((data, (row, col)), shape=(adj_list.max() + 1, adj_list.max() + 1), dtype=np.int8)
        print "Adjacency Matrix created"
        # for e in adj_list:
        #     adj_mat[e[0], e[1]] = 1
        #     adj_mat[e[1], e[0]] = 1

        # self.G = nx.from_numpy_matrix(adj_mat)
        self.G = nx.from_scipy_sparse_matrix(adj_mat)
        print "NetworkX Graph created"
        self.total_num_nodes = self.G.number_of_nodes()
        self.total_num_edges = self.G.number_of_edges()


    @profiler
    def perform_slash_burn(self, k, gcc_num_nodes_criterion=7):
        """ Peforms SlashBurn algorithm for subgraph generation
        
        Args:
            k: number of hubsets to remove at each iteration
            gcc_num_nodes_criterion: the inclusive upper-bound criterion for a subgraph to be a GCC which will be burned
        """

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
                self.process_subgraph(current_gcc.subgraph(hubset_subgraph))

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
                    self.process_subgraph(sub_graph)
                else:
                    # append the subgraph to GCCs queue
                    gcc_queue.append(sub_graph)
                # add the nodes in the non-GCC to the back of gamma
                self.gamma = np.append(self.gamma, sub_graph.nodes())

        if self.parallel:
            self.workers.close()
            self.workers.join()

    def process_subgraph(self, sub_graph):
        if self.parallel:
            self.workers.apply_async(mdl_encoding,
                                     args=(sub_graph, self.total_num_nodes),
                                     callback=self.collect_results)
        else:
            self.collect_results(mdl_encoding(sub_graph, self.total_num_nodes))

    def collect_results(self, result):
        # TODO: handle race conditions here!!!
	lock.acquire()
        if len(self.top_k_structures) < self.top_k:
            print "Adding", result.__class__.__name__
            heapq.heappush(self.top_k_structures, (result.benefit, result))
        else:
            print result.__class__.__name__, result.benefit, self.top_k_structures[0][0]
            if self.top_k_structures[0][0] < result.benefit:
                print "Adding", result.__class__.__name__, \
                    "and removing", self.top_k_structures[0][1].__class__.__name__
                heapq.heappushpop(self.top_k_structures, (result.benefit, result))
	lock.release()

def mdl_encoding(sub_graph, total_num_nodes):
    # try:
        err = structures.Error(sub_graph, total_num_nodes)
        err.compute_mdl_cost()
        structure_types = [
            structures.Chain(sub_graph, total_num_nodes),
            structures.Clique(sub_graph, total_num_nodes),
            structures.Star(sub_graph, total_num_nodes),
            structures.BipartiteCore(sub_graph, total_num_nodes),
        ]
        for st in structure_types:
            st.compute_mdl_cost()
            st.benefit = err.mdl_cost - st.mdl_cost
        err.benefit = 0
        structure_types.append(err)
        optimal_structure = min(structure_types, key=lambda k: k.mdl_cost)
        return optimal_structure
    # except:
        # Put all exception text into an exception and raise that
        # raise Exception("".join(traceback.format_exception(*sys.exc_info())))

if __name__ == '__main__':
    vog = VoG('./soc-Epinions1.txt', time_limit=600, parallel=True)

