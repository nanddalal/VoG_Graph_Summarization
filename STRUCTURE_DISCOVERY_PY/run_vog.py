import os
import sys

from summarization import VoG


if __name__ == '__main__':
    output_file_name = sys.argv[1]
    subgraph_generation_algo = sys.argv[2]
    hyperparameter1 = int(sys.argv[3])
    hyperparameter2 = int(sys.argv[4])

    vog = VoG('../DATA/flickr/flickr.graph', delimiter=',', zero_indexed=False,
              output_file=output_file_name,
              subgraph_generation_algo=subgraph_generation_algo,
              hubset_k=hyperparameter1,
              gcc_num_nodes_criterion=hyperparameter2,
              min_egonet_size=hyperparameter1,
              egonet_num_nodes_criterion=hyperparameter2)

    os.system('python ../MDL/score.py ./modified_edge_file.txt ' + output_file_name +
              ' > ' + output_file_name+'.lgm')

