import os
from summarization import VoG


if __name__ == '__main__':

    experimental_results_path = './experimental_results/'
    if not os.path.exists(experimental_results_path):
        os.makedirs(experimental_results_path)

    subgraph_generation_algos = ['k_hop_egonets', 'modified_slash_burn']
    min_egonet_sizes = [10, 20, 50, 100]
    egonet_num_nodes_criterions = [50, 100, 200, 500, 1000]
    hubset_ks = [1, 4, 8]
    gcc_num_nodes_criterions = [50, 100, 200]

    for subgraph_generation_algo in subgraph_generation_algos:
        if subgraph_generation_algo == 'modified_slash_burn':
            for hubset_k in hubset_ks:
                for gcc_num_nodes_criterion in gcc_num_nodes_criterions:
                    output_file_name = experimental_results_path + \
                                       subgraph_generation_algo + '_' + \
                                       str(hubset_k) + '_' + \
                                       str(gcc_num_nodes_criterion) + \
                                       '.out'
                    vog = VoG('../DATA/flickr/flickr.graph', delimiter=',', zero_indexed=False,
                              output_file=output_file_name,
                              subgraph_generation_algo=subgraph_generation_algo,
                              hubset_k=hubset_k,
                              gcc_num_nodes_criterion=gcc_num_nodes_criterion)
                    os.system('python ../MDL/score.py ./modified_edge_file.txt ' + output_file_name +
                              ' > ' + output_file_name+'.lgm')
        elif subgraph_generation_algo == 'k_hop_egonets':
            for min_egonet_size in min_egonet_sizes:
                for egonet_num_nodes_criterion in egonet_num_nodes_criterions:
                    if egonet_num_nodes_criterion > min_egonet_size:
                        output_file_name = experimental_results_path + \
                                           subgraph_generation_algo + '_' + \
                                           str(min_egonet_size) + '_' + \
                                           str(egonet_num_nodes_criterion) + \
                                           '.out'
                        vog = VoG('../DATA/flickr/flickr.graph', delimiter=',', zero_indexed=False,
                                  output_file=output_file_name,
                                  subgraph_generation_algo=subgraph_generation_algo,
                                  min_egonet_size=min_egonet_size,
                                  egonet_num_nodes_criterion=egonet_num_nodes_criterion)
                        os.system('python ../MDL/score.py ./modified_edge_file.txt ' + output_file_name +
                                  ' > ' + output_file_name+'.lgm')

