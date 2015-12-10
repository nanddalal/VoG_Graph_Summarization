import os
import subprocess as sp


if __name__ == '__main__':

    experimental_results_path = './experimental_results/'
    if not os.path.exists(experimental_results_path):
        os.makedirs(experimental_results_path)

    # subgraph_generation_algos = ['k_hop_egonets', 'modified_slash_burn']
    subgraph_generation_algos = ['k_hop_egonets']
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
                    print "LAUNCHING"
                    sp.Popen(['ipython', 'run_vog.py',
                              output_file_name, subgraph_generation_algo,
                              str(hubset_k), str(gcc_num_nodes_criterion)])
        elif subgraph_generation_algo == 'k_hop_egonets':
            for min_egonet_size in min_egonet_sizes:
                for egonet_num_nodes_criterion in egonet_num_nodes_criterions:
                    if egonet_num_nodes_criterion > min_egonet_size:
                        output_file_name = experimental_results_path + \
                                           subgraph_generation_algo + '2_' + \
                                           str(min_egonet_size) + '_' + \
                                           str(egonet_num_nodes_criterion) + \
                                           '.out'
                        print "LAUNCHING"
                        sp.Popen(['ipython', 'run_vog.py',
                                  output_file_name, subgraph_generation_algo,
                                  str(min_egonet_size), str(egonet_num_nodes_criterion)])

