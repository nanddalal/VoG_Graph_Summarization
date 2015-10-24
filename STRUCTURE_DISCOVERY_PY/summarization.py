from operator import itemgetter
import csv
import numpy as np
import networkx as nx

class VoG:

    # this assumes a 1 indexed adjacency list
    def parse_adj_list_file(self, file_name):
        with open(file_name) as gf:
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

    # def mdl_encoding(self):
        # TODO: perform MDL encodings

if __name__ == '__main__':
    vog = VoG()
    vog.parse_adj_list_file('sb_paper_graph.txt')
    vog.slash_burn(k=2)
