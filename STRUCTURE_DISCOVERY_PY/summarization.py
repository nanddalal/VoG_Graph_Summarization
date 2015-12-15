#!/usr/bin/env python

import os
import time
import signal
import csv
import heapq
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
import multiprocessing as mp

from subgraph_generation_algos import modified_slash_burn, k_hop_egonets


class VoGTimeout(Exception):
    @staticmethod
    def time_limit_handler(signum, frame):
        print "Waited enough time for the workers to join"
        raise VoGTimeout


class VoG:
    def __init__(self, dataset, input_dir, input_fn, delimiter, zero_indexed,
                 subgraph_generation_algo,
                 hubset_k=1, gcc_num_nodes_criterion=100,
                 min_egonet_size=10, egonet_num_nodes_criterion=100, hop_k=1,
                 top_k=100, num_iterations=20):

        if subgraph_generation_algo == 'modified_slash_burn':
            self.model_file = './' + dataset + \
                              '_' + subgraph_generation_algo + \
                              '_' + 'top' + str(top_k) + \
                              '_' + 'iter' + str(num_iterations) + \
                              '_' + 'hyperparams' + \
                              '_' + str(hubset_k) + \
                              '_' + str(gcc_num_nodes_criterion)
        elif subgraph_generation_algo == 'k_hop_egonets':
            self.model_file = './' + dataset + \
                              '_' + subgraph_generation_algo + \
                              '_' + 'top' + str(top_k) + \
                              '_' + 'iter' + str(num_iterations) + \
                              '_' + 'hyperparams' + \
                              '_' + str(min_egonet_size) + \
                              '_' + str(egonet_num_nodes_criterion) + \
                              '_' + str(hop_k)

        self.dataset = dataset
        self.input_dir = input_dir
        self.input_fn = input_fn
        self.delimiter = delimiter
        self.zero_indexed = zero_indexed

        self.subgraph_generation_algo = subgraph_generation_algo

        self.hubset_k = hubset_k
        self.gcc_num_nodes_criterion = gcc_num_nodes_criterion

        self.min_egonet_size = min_egonet_size
        self.egonet_num_nodes_criterion = egonet_num_nodes_criterion
        self.hop_k = hop_k

        self.top_k = top_k
        self.num_iterations = num_iterations

        self.top_k_structures = []
        # print "Printing top k structures"
        # self.print_top_k_structures()

    def __str__(self):
        return self.model_file

    def print_top_k_structures(self):
        # Sorting top k structures using heap sort
        self.top_k_structures.sort(reverse=True)
        for s in self.top_k_structures:
            print s[1].__class__.__name__, s[1].graph.nodes()

    @staticmethod
    def read_adj_list_file(input_dir, input_fn, delimiter, zero_indexed):
        with open(input_dir+input_fn) as gf:
            r = csv.reader(gf, delimiter=delimiter)
            adj_list = np.array(list(r), int)

        if not zero_indexed:
            adj_list -= 1

        return adj_list

    @staticmethod
    def create_normalized_file(dataset, input_dir, input_fn, delimiter, zero_indexed):
        adj_list = VoG.read_adj_list_file(input_dir, input_fn, delimiter, zero_indexed)

        normalize_fn = input_dir+dataset+'.normalized'
        with open(normalize_fn, 'w') as nf:
            for e in adj_list:
                nf.write("%s,%s\n" % (e[0]+1, e[1]+1))

        return normalize_fn

    def construct_graph_from_adj_list(self, adj_list):
        rows = [e[0] for e in adj_list]
        cols = [e[1] for e in adj_list]

        print "Constructing adjacency matrix"
        adj_mat = csr_matrix((np.ones(len(adj_list)), (rows, cols)),
                             shape=(adj_list.max() + 1, adj_list.max() + 1))
        adj_mat = adj_mat + adj_mat.T

        print "Constructing NetworkX graph"
        self.G = nx.from_scipy_sparse_matrix(adj_mat)

        self.total_num_nodes = self.G.number_of_nodes()
        self.total_num_edges = self.G.number_of_edges()

    def summarize(self):

        print "Constructing graph"
        adj_list = VoG.read_adj_list_file(self.input_dir, self.input_fn, self.delimiter, self.zero_indexed)
        self.construct_graph_from_adj_list(adj_list)

        # Initializing the shared top k queue
        self.manager = mp.Manager()
        self.top_k_queue = self.manager.JoinableQueue()

        # Lock and CV for handling the subgraph queue and stopping criterion
        self.subgraph_queue_lock = mp.Lock()
        self.subgraph_queue_cv = mp.Condition(self.subgraph_queue_lock)

        print "Initializing", mp.cpu_count(), "workers"
        self.workers = mp.Pool(processes=None)

        start_time = time.time()
        print "Performing graph summarization using top k heuristic"
        self.perform_graph_summarization(self.hubset_k, self.gcc_num_nodes_criterion,
                                         self.min_egonet_size, self.egonet_num_nodes_criterion, self.hop_k)
        end_time = time.time()

        print "Shutting down manager and attempting to terminate/join the workers"
        signal.signal(signal.SIGALRM, VoGTimeout.time_limit_handler)
        signal.alarm(10)
        try:
            self.manager.shutdown()
            self.workers.terminate()
            self.workers.join()
        except Exception as e:
            print "Got exception while terminating workers", e
            signal.alarm(0)
        else:
            print "Workers joined - resetting the SIGALRM"
            signal.alarm(0)

        return end_time - start_time

    def perform_graph_summarization(self,
                                    hubset_k, gcc_num_nodes_criterion,
                                    min_egonet_size, egonet_num_nodes_criterion, hop_k):

        print "Spinning off the update top k process to handle assembly of the top k structures"
        self.workers.apply_async(update_top_k,
                                 args=(self.top_k_queue, self.top_k, self.model_file),
                                 callback=self.collect_top_k_structures)

        print "Initializing the subgraph queue to be the whole graph"
        self.subgraph_queue = [self.G]

        iteration = 0
        self.num_finished = 0
        self.num_to_finish = 1

        print "Entering main subgraph generation loop along with MDL encodings"
        while True:
            self.subgraph_queue_lock.acquire()

            # Wait until the subgraph queue becomes non empty
            while len(self.subgraph_queue) <= 0:
                self.subgraph_queue_cv.wait()

            # The processed subgraphs will be updating these values so that we stop when we are at the appropriate depth
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
                                                   hop_k,
                                                   self.total_num_nodes,
                                                   self.top_k_queue,
                                                   iteration),
                                             callback=self.collect_resulting_subgraphs)
            self.subgraph_queue = []

            iteration += 1

            self.subgraph_queue_lock.release()

        print "We had to process", self.num_to_finish, "and we processed", self.num_finished, "subgraphs"

    def collect_top_k_structures(self, top_k_structs):
        self.top_k_structures = top_k_structs

    def collect_resulting_subgraphs(self, iteration_subgraphs):
        iteration = iteration_subgraphs[0]
        subgraphs = iteration_subgraphs[1]

        self.subgraph_queue_lock.acquire()

        self.subgraph_queue += subgraphs
        self.num_finished += 1
        # Kind of like doing iterative deepening?
        # We will keep increasing the number of subgraphs we have to process until we reach the specified num iterations
        if iteration < self.num_iterations:
            self.num_to_finish += len(subgraphs)

        self.subgraph_queue_cv.notify()

        self.subgraph_queue_lock.release()


def update_top_k(top_k_queue, top_k, model_file):
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

        # Every 10 iterations, save the top k structures to disk
        if iteration % 10 == 0:
            with open(model_file, 'w') as out:
                for s in top_k_structs:
                    try:
                        out.write(str(s[1]) + '\n')
                    except NotImplementedError:
                        pass
        iteration += 1

    return top_k_structs


if __name__ == '__main__':
    kwargs = {
        'dataset': 'flickr',
        'input_dir': '../DATA/flickr/',
        'input_fn': 'flickr.graph',
        'delimiter': ',',
        'zero_indexed': False
    }
    normalized_fn = VoG.create_normalized_file(**kwargs)
    vog = VoG(subgraph_generation_algo='k_hop_egonets', **kwargs)
    print "LAUNCHING", str(vog)
    runtime = vog.summarize()
    print "RUNTIME:", runtime
    os.system('python ../MDL/score.py ' + normalized_fn + ' ' + str(vog) +
              ' > ' + str(vog)+'.lgm')
    os.system('echo ' + str(runtime) + ' >> ' + str(vog)+'.lgm')

