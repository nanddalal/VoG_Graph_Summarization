#!/usr/bin/env python

import os
import time
import csv
import heapq
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
import multiprocessing as mp
# import matplotlib.pyplot as plt

from subgraph_generation_algos import modified_slash_burn, k_hop_egonets


class VoG:
    def __init__(self, input_file, delimiter, zero_indexed, output_file,
                 subgraph_generation_algo,
                 hubset_k=1, gcc_num_nodes_criterion=7,
                 min_egonet_size=40, egonet_num_nodes_criterion=200,
                 top_k=100, num_iterations=20):

        self.output_file = output_file

        self.subgraph_generation_algo = subgraph_generation_algo
        self.num_iterations = num_iterations

        self.top_k = top_k
        self.top_k_structures = []

        self.subgraph_queue_lock = mp.Lock()
        self.subgraph_queue_cv = mp.Condition(self.subgraph_queue_lock)

        print "Parsing adjacency list"
        self.parse_adj_list_file(input_file, delimiter, zero_indexed)
        # self.visualize_graph()

        self.manager = mp.Manager()
        self.top_k_queue = self.manager.JoinableQueue()
        self.workers = mp.Pool(processes=None)

        print "Performing subgraph generation and labeling using top k heuristic"
        self.perform_graph_summarization(hubset_k, gcc_num_nodes_criterion, min_egonet_size, egonet_num_nodes_criterion)

        print "Shutting down manager and terminating/joining the workers"
        self.manager.shutdown()
        time.sleep(5)
        # try:
        #     self.workers.terminate()
        #     self.workers.join()
        # except Exception as e:
        #     print "Got exception while terminating workers", e

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

    def parse_adj_list_file(self, input_file, delimiter, zero_indexed):
        with open(input_file) as gf:
            r = csv.reader(gf, delimiter=delimiter)
            adj_list = np.array(list(r), int)

        if not zero_indexed:
            adj_list -= 1

        row, col, data = [], [], []
        with open("./modified_edge_file.txt", 'w') as modified_edge_file:
            for e in adj_list:
                row.append(e[0])
                col.append(e[1])
                data.append(1)
                modified_edge_file.write("%s,%s\n" % (e[0]+1, e[1]+1))

        print "Constructing adjacency matrix"
        adj_mat = csr_matrix((data, (row, col)), shape=(adj_list.max() + 1, adj_list.max() + 1), dtype=np.int8)

        print "Constructing networkx graph"
        self.G = nx.from_scipy_sparse_matrix(adj_mat)

        self.total_num_nodes = self.G.number_of_nodes()
        self.total_num_edges = self.G.number_of_edges()

    # @profiler
    def perform_graph_summarization(self,
                                    hubset_k, gcc_num_nodes_criterion,
                                    min_egonet_size, egonet_num_nodes_criterion):
        """ Peforms graph summarization
        
        Args:
            k: number of hubsets to remove at each iteration
            gcc_num_nodes_criterion: the inclusive upper-bound criterion for a subgraph to be a GCC which will be burned
        """

        self.workers.apply_async(update_top_k,
                                 args=(self.top_k_queue, self.top_k, self.output_file),
                                 callback=self.collect_top_k_structures)

        self.subgraph_queue = [self.G]

        iteration = 0
        self.num_finished = 0
        self.num_to_finish = 1

        while True:
            self.subgraph_queue_lock.acquire()

            while len(self.subgraph_queue) <= 0:
                self.subgraph_queue_cv.wait()

            if self.num_finished >= self.num_to_finish:
                break

            print "Spinning off subgraph generation for", len(self.subgraph_queue), "subgraphs"
            for subgraph in self.subgraph_queue:
                if self.subgraph_generation_algo == 'modified_slash_burn':
                    self.workers.apply_async(modified_slash_burn,
                                             args=(subgraph,
                                                   hubset_k,
                                                   gcc_num_nodes_criterion,
                                                   self.total_num_nodes,
                                                   self.top_k_queue,
                                                   iteration),
                                             callback=self.collect_resulting_subgraphs)
                elif self.subgraph_generation_algo == 'k_hop_egonets':
                    self.workers.apply_async(k_hop_egonets,
                                             args=(subgraph,
                                                   min_egonet_size,
                                                   egonet_num_nodes_criterion,
                                                   self.total_num_nodes,
                                                   self.top_k_queue,
                                                   iteration),
                                             callback=self.collect_resulting_subgraphs)
            self.subgraph_queue = []

            iteration += 1

            self.subgraph_queue_lock.release()

        print "We had to finish", self.num_to_finish, "and we finished", self.num_finished

    def collect_top_k_structures(self, top_k_structs):
        print "Called update top k's calback function"
        self.top_k_structures = top_k_structs

    def collect_resulting_subgraphs(self, iteration_subgraphs):
        iteration = iteration_subgraphs[0]
        subgraphs = iteration_subgraphs[1]

        self.subgraph_queue_lock.acquire()

        self.subgraph_queue += subgraphs
        self.num_finished += 1
        if iteration < self.num_iterations:
            self.num_to_finish += len(subgraphs)

        self.subgraph_queue_cv.notify()

        self.subgraph_queue_lock.release()


def update_top_k(top_k_queue, top_k, output_file):
    iteration = 0
    top_k_structs = []
    while True:
        try:
            structure = top_k_queue.get()
        except Exception as e:
            print "Manager was shut down so should be calling callback", e
            break
        if len(top_k_structs) < top_k:
            print "Adding", structure.__class__.__name__
            heapq.heappush(top_k_structs, (structure.benefit, structure))
        else:
            if top_k_structs[0][0] < structure.benefit:
                print "Adding", structure.__class__.__name__, \
                    "and removing", top_k_structs[0][1].__class__.__name__
                heapq.heappushpop(top_k_structs, (structure.benefit, structure))
        if iteration % 10 == 0:
            with open(output_file, 'w') as out:
                for s in top_k_structs:
                    try:
                        out.write(str(s[1]) + '\n')
                    except NotImplementedError:
                        pass
        iteration += 1

    return top_k_structs


if __name__ == '__main__':
    vog = VoG('../DATA/flickr/flickr.graph', delimiter=',', zero_indexed=False, output_file='lol.txt',
              subgraph_generation_algo='k_hop_egonets')
    os.system('python ../MDL/score.py ./modified_edge_file.txt ' + 'lol.txt' + ' > ' + 'lol.txt'+'.lgm')

