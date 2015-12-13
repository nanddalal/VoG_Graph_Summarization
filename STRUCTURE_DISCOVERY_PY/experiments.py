import os
import subprocess as sp

from summarization import VoG


if __name__ == '__main__':

    kwargs = {
        'dataset': 'flickr',
        'input_dir': '../DATA/flickr/',
        'input_fn': 'flickr.graph',
        'delimiter': ',',
        'zero_indexed': False
    }
    normalized_fn = VoG.create_normalized_file(**kwargs)

    subgraph_generation_algos = ['k_hop_egonets', 'modified_slash_burn']
    hubset_ks = [8, 4, 1]
    gcc_num_nodes_criterions = [50, 100, 200]
    min_egonet_sizes = [10, 20, 50, 100]
    egonet_num_nodes_criterions = [50, 100, 200, 500, 1000]
    hop_ks = [1, 2]

    for subgraph_generation_algo in subgraph_generation_algos:
        if subgraph_generation_algo == 'modified_slash_burn':
            for hubset_k in hubset_ks:
                for gcc_num_nodes_criterion in gcc_num_nodes_criterions:
                    print "LAUNCHING"
                    sp.Popen(['ipython', 'run_vog.py',
                              normalized_fn, subgraph_generation_algo,
                              str(hubset_k), str(gcc_num_nodes_criterion), str(1)])
        elif subgraph_generation_algo == 'k_hop_egonets':
            for hop_k in hop_ks:
                for min_egonet_size in min_egonet_sizes:
                    for egonet_num_nodes_criterion in egonet_num_nodes_criterions:
                        if egonet_num_nodes_criterion > min_egonet_size:
                            print "LAUNCHING"
                            sp.Popen(['ipython', 'run_vog.py',
                                      normalized_fn, subgraph_generation_algo,
                                      str(min_egonet_size), str(egonet_num_nodes_criterion), str(hop_k)])

