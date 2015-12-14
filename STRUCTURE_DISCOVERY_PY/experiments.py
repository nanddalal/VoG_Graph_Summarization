import os

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
    min_egonet_sizes = [10, 50, 100]
    egonet_num_nodes_criterions = [100, 200, 500]
    hop_ks = [2, 1]

    for subgraph_generation_algo in subgraph_generation_algos:
        if subgraph_generation_algo == 'modified_slash_burn':
            for hubset_k in hubset_ks:
                for gcc_num_nodes_criterion in gcc_num_nodes_criterions:
                    vog = VoG(subgraph_generation_algo=subgraph_generation_algo,
                              hubset_k=hubset_k,
                              gcc_num_nodes_criterion=gcc_num_nodes_criterion,
                              **kwargs)
                    print "LAUNCHING", str(vog)
                    runtime = vog.summarize()
                    print "RUNTIME:", runtime
                    os.system('python ../MDL/score.py ' + normalized_fn + ' ' + str(vog) +
                              ' > ' + str(vog)+'.lgm')
                    os.system('echo ' + str(runtime) + ' >> ' + str(vog)+'.lgm')
        elif subgraph_generation_algo == 'k_hop_egonets':
            for hop_k in hop_ks:
                for min_egonet_size in min_egonet_sizes:
                    for egonet_num_nodes_criterion in egonet_num_nodes_criterions:
                        if egonet_num_nodes_criterion > min_egonet_size:
                            vog = VoG(subgraph_generation_algo=subgraph_generation_algo,
                                      min_egonet_size=min_egonet_size,
                                      egonet_num_nodes_criterion=egonet_num_nodes_criterion,
                                      hop_k=hop_k,
                                      **kwargs)
                            print "LAUNCHING", str(vog)
                            runtime = vog.summarize()
                            print "RUNTIME:", runtime
                            os.system('python ../MDL/score.py ' + normalized_fn + ' ' + str(vog) +
                                      ' > ' + str(vog)+'.lgm')
                            os.system('echo ' + str(runtime) + ' >> ' + str(vog)+'.lgm')

