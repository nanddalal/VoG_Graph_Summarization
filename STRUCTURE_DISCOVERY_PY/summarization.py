from operator import itemgetter
import csv
import numpy as np
import networkx as nx
import multiprocessing as mp

class VoG:
    def __init__(self, input_file):
        self.mdl_results = []
        self.parse_adj_list_file(input_file)
        self.mdl_workers = mp.Pool(processes=(mp.cpu_count() * 2))
        self.slash_burn_new(k=2, gcc_num_nodes_criterion=3)


    # this assumes a 1 indexed adjacency list
    def parse_adj_list_file(self, input_file):
        with open(input_file) as gf:
            r = csv.reader(gf, delimiter=',')
            self.adj_list = np.array(list(r), int)

        self.adj_mat = np.zeros((self.adj_list.max(), self.adj_list.max()))
        self.adj_list -= 1 # TODO: fix this!!!
        for e in self.adj_list:
            self.adj_mat[e[0], e[1]] = 1
            self.adj_mat[e[1], e[0]] = 1


        self.G = nx.from_numpy_matrix(self.adj_mat)

    def generate_sub_graph(self, node_list):
        return self.G.subgraph(node_list)

    def slash_burn(self, k):
        self.gamma = np.array([]) # deque
        self.candidate_structures = []

        while self.G.number_of_nodes() > k:
            # 1
            k_hubset_nd = sorted(self.G.degree_iter(), key=itemgetter(1), reverse=True)[0:k] # get a sorted list of (node, degree) in decreasing order
            k_hubset = [i[0] for i in k_hubset_nd] # get the node index for the k highest degree vertex
            self.G.remove_nodes_from(k_hubset) # remove the k hubset from G, so now we have G' (slash!)
            self.gamma = np.insert(self.gamma, 0, k_hubset) # add removed k hubset to the front of gamma

            # 2
            sorted_sub_graphs = [(sub_graph, sub_graph.number_of_nodes()) for sub_graph in nx.connected_component_subgraphs(self.G)] # get all the subgraphs after removing the k hubset
            sorted_sub_graphs = sorted(sorted_sub_graphs, key=itemgetter(1), reverse=True) # sort the subgraphs by the number of nodes in decreasing order

            GCC = sorted_sub_graphs[0][0] # because we have sorted, the zeroth subgraph is the greatest connected component
            for sub_graph, num_nodes in sorted_sub_graphs[1:]: # iterate over the remaining subgraphs we are "burning"
                self.candidate_structures.append(sub_graph)
                self.gamma = np.append(self.gamma, sub_graph.nodes()) # add the nodes in the non-GCC to the back of gamma

            # 3
            self.G = GCC

    def slash_burn_new(self, k, gcc_num_nodes_criterion=3):
        """ Peforms slashburn algorithm for subgraph generation
        
        Args:
            k: number of hubsets to remove at each iteration
            gcc_num_nodes_criterion: the criterion for a subgraph to be a GCC which will be burned
        """
        self.GCCs = [self.G] # all the GCCs that meet the criterion, acts as a queue
        self.gamma = np.array([]) # deque
        self.candidate_structures = []

        while len(self.GCCs) > 0:
            curr_GCC = self.GCCs[0]; # extract the first element of self.GCCs
            self.GCCs = self.GCCs[1:]; # remove the first element of self.GCCs
            # 1
            k_hubset_nd = sorted(curr_GCC.degree_iter(), key=itemgetter(1), reverse=True)[0:k] # get a sorted list of (node, degree) in decreasing order
            k_hubset = [i[0] for i in k_hubset_nd] # get the node index for the k highest degree vertex
            curr_GCC.remove_nodes_from(k_hubset) # remove the k hubset from G, so now we have G' (slash!)
            self.gamma = np.insert(self.gamma, 0, k_hubset) # add removed k hubset to the front of gamma

            # 2
            sorted_sub_graphs = [(sub_graph, sub_graph.number_of_nodes()) for sub_graph in nx.connected_component_subgraphs(curr_GCC)] # get all the subgraphs after removing the k hubset
            sorted_sub_graphs = sorted(sorted_sub_graphs, key=itemgetter(1), reverse=True) # sort the subgraphs by the number of nodes in decreasing order

            for sub_graph, num_nodes in sorted_sub_graphs: # iterate over the remaining subgraphs we are "burning"
                self.mdl_workers.apply_async(mdl_encoding, args=sub_graph, callback=self.collect_mdl_results)
                if sub_graph.number_of_nodes() <= gcc_num_nodes_criterion:
                    self.candidate_structures.append(sub_graph) # meets the criterion, goes to candidate structures
                else:
                    self.GCCs.append(sub_graph) # append the subgraph to GCCs queue
                self.gamma = np.append(self.gamma, sub_graph.nodes()) # add the nodes in the non-GCC to the back of gamma

        print
        self.mdl_workers.close()
        self.mdl_workers.join()

    def collect_mdl_results(self, result):
        self.mdl_results.append(result)

def mdl_encoding(self, sub_graph):
    return sub_graph
    # TODO: perform MDL encodings


if __name__ == '__main__':
    vog = VoG('sb_paper_graph.txt')
    # vog.slash_burn(k=2)