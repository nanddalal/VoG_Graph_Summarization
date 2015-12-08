#!/usr/bin/env python

import sys
import signal
import csv
import heapq
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
import multiprocessing as mp
# import matplotlib.pyplot as plt
from operator import itemgetter

import structures
from profiling import profiler


class VoGTimeout(Exception):
    pass


class VoG:
    def __init__(self, input_file, delimiter="\t", zero_indexed=True,  hubset_k=1, gcc_num_nodes_criterion=7, top_k=10, time_limit=None):
        self.top_k = top_k
        self.top_k_structures = []

        self.gcc_queue_lock = mp.Lock()
        self.gcc_queue_cv = mp.Condition(self.gcc_queue_lock)

        print "Parsing adjacency list"
        self.parse_adj_list_file(input_file, delimiter=delimiter)
        # self.visualize_graph()

        if time_limit is not None:
            signal.signal(signal.SIGALRM, VoG.time_limit_handler)
            signal.alarm(time_limit)

        self.manager = mp.Manager()
        self.top_k_queue = self.manager.JoinableQueue()
        self.workers = mp.Pool(processes=None)

        try:
            print "Performing slash burn using top k heuristic"
            # self.perform_slash_burn(slash_burn_k, int(math.log(self.total_num_nodes)))
            self.perform_slash_burn(hubset_k, gcc_num_nodes_criterion)
        except VoGTimeout:
            pass  # TODO: probably should be doing something here
        else:
            signal.alarm(0)  # TODO: understand why this is necessary

        print "Shutting down manager and terminating/joining the workers"
        self.manager.shutdown()
        self.workers.terminate()
        self.workers.join()

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

    @staticmethod
    def time_limit_handler(signum, frame):
        print "Reached specified time limit"
        raise VoGTimeout

    def print_top_k_structures(self):
        self.top_k_structures.sort(reverse=True)
        for s in self.top_k_structures:
            print s[1].__class__.__name__, s[1].graph.nodes()

    # TODO: this assumes a 1 indexed adjacency list
    # should only have to change the delimiter and whether to minus-equal one
    def parse_adj_list_file(self, input_file, zero_indexed=True, delimiter="\t"):
        with open(input_file) as gf:
            r = csv.reader(gf, delimiter=delimiter)
            adj_list = np.array(list(r), int)

        if not zero_indexed:
            adj_list -= 1
        row, col, data = [], [], []
        for e in adj_list:
            row.append(e[0])
            col.append(e[1])
            data.append(1)

        adj_mat = csr_matrix((data, (row, col)), shape=(adj_list.max() + 1, adj_list.max() + 1), dtype=np.int8)
        print "Adjacency Matrix created"

        self.G = nx.from_scipy_sparse_matrix(adj_mat)
        print "NetworkX Graph created"

        self.total_num_nodes = self.G.number_of_nodes()
        self.total_num_edges = self.G.number_of_edges()

    # @profiler
    def perform_slash_burn(self, hubset_k, gcc_num_nodes_criterion):
        """ Peforms SlashBurn algorithm for subgraph generation
        
        Args:
            k: number of hubsets to remove at each iteration
            gcc_num_nodes_criterion: the inclusive upper-bound criterion for a subgraph to be a GCC which will be burned
        """

        self.workers.apply_async(update_top_k,
                                 args=(self.top_k_queue, self.top_k),
                                 callback=self.collect_top_k_structures)

        self.gcc_queue = [self.G]

        while True:  # TODO: come up with stopping criterion other than time
            self.gcc_queue_lock.acquire()

            while len(self.gcc_queue) <= 0:
                self.gcc_queue_cv.wait()

            print "Spinning off slash burns for", len(self.gcc_queue), "gccs"
            for gcc in self.gcc_queue:
                self.workers.apply_async(slash_and_burn,
                                         args=(gcc,
                                               hubset_k,
                                               gcc_num_nodes_criterion,
                                               self.total_num_nodes,
                                               self.top_k_queue),
                                         callback=self.collect_slashburned_gccs)
            self.gcc_queue = []

            self.gcc_queue_lock.release()

    def collect_top_k_structures(self, top_k_structs):
        self.top_k_structures = top_k_structs

    def collect_slashburned_gccs(self, gccs):
        self.gcc_queue_lock.acquire()
        self.gcc_queue += gccs
        self.gcc_queue_cv.notify()
        self.gcc_queue_lock.release()


def update_top_k(top_k_queue, top_k):
    top_k_structs = []
    while True:
        try:
            structure = top_k_queue.get()
        except EOFError as eof:
            break
        if len(top_k_structs) < top_k:
            print "Adding", structure.__class__.__name__
            heapq.heappush(top_k_structs, (structure.benefit, structure))
        else:
            if top_k_structs[0][0] < structure.benefit:
                print "Adding", structure.__class__.__name__, \
                    "and removing", top_k_structs[0][1].__class__.__name__
                heapq.heappushpop(top_k_structs, (structure.benefit, structure))
    return top_k_structs


def slash_and_burn(current_gcc, hubset_k, gcc_num_nodes_criterion, total_num_nodes, top_k_queue):
    gccs = []

    print "Finding k hubset", current_gcc.number_of_nodes(), current_gcc.number_of_edges()
    # 1
    # get a sorted list of (node, degree) in decreasing order
    k_hubset_nd = sorted(current_gcc.degree_iter(), key=itemgetter(1), reverse=True)
    # get the node index for the k highest degree vertex
    k_hubset = [i[0] for i in k_hubset_nd[0:hubset_k]]

    # consider subgraphs (stars) consisting of centrality nodes from the k hubset we are about to slash
    for node in k_hubset:
        hubset_subgraph = current_gcc.neighbors(node)
        hubset_subgraph.append(node)
        structure = mdl_encoding(current_gcc.subgraph(hubset_subgraph), total_num_nodes)
        try:
            top_k_queue.put(structure)
        except EOFError as eof:
            pass

    # remove the k hubset from G, so now we have G' (slash!)
    current_gcc.remove_nodes_from(k_hubset)

    print "Finding remaining subgraphs after having removed k hubset"
    # 2
    # get all the subgraphs after removing the k hubset
    sub_graphs = nx.connected_component_subgraphs(current_gcc, copy=False)

    print "Iterating over remaining subgraphs and spinning off labeling if less than certain size"
    # iterate over the remaining subgraphs we are "burning"
    for sub_graph in sub_graphs:
        if sub_graph.number_of_nodes() <= gcc_num_nodes_criterion:
            structure = mdl_encoding(sub_graph, total_num_nodes)
            try:
                top_k_queue.put(structure)
            except EOFError as eof:
                pass
        else:
            # append the subgraph to GCCs queue
            gccs.append(sub_graph)

    return gccs


def mdl_encoding(sub_graph, total_num_nodes):
    err = structures.Error(sub_graph)
    err.compute_mdl_cost()
    structure_types = [
        structures.Clique(sub_graph, total_num_nodes),
        structures.Star(sub_graph, total_num_nodes),
        structures.BipartiteCore(sub_graph, total_num_nodes),
        structures.NearBipartiteCore(sub_graph, total_num_nodes),
        # structures.Chain(sub_graph, total_num_nodes),
    ]
    print sub_graph.nodes(), sub_graph.edges()
    for st in structure_types:
        st.compute_mdl_cost()
        st.benefit = err.mdl_cost - st.mdl_cost
        print st.__class__.__name__, st.mdl_cost, st.benefit
    err.benefit = 0
    print err.__class__.__name__, err.mdl_cost, err.benefit
    structure_types.append(err)
    optimal_structure = min(structure_types, key=lambda k: k.mdl_cost)
    print "Picked", optimal_structure.__class__.__name__, optimal_structure.mdl_cost, optimal_structure.benefit
    return optimal_structure


def debug_print(debug):
    print debug
    sys.stdout.flush()

if __name__ == '__main__':
    # vog = VoG('./test_cliqueStarBCChain.txt', hubset_k=1, gcc_num_nodes_criterion=5, top_k=10, time_limit=15)
    vog = VoG('../DATA/test_100000.txt', delimiter="\t", zero_indexed=True, hubset_k=1, gcc_num_nodes_criterion=5, top_k=10, time_limit=300)
    # vog = VoG('./test_cliqueStarBCChain.txt', delimiter=",", zero_indexed=False, hubset_k=1, gcc_num_nodes_criterion=5, top_k=10, time_limit=15)