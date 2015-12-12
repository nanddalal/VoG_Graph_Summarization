import os
import sys

from summarization import VoG


if __name__ == '__main__':
    normalized_fn = sys.argv[1]
    subgraph_generation_algo = sys.argv[2]
    hyperparam1 = int(sys.argv[3])
    hyperparam2 = int(sys.argv[4])
    hyperparam3 = int(sys.argv[5])

    kwargs = {
        'dataset': 'flickr',
        'input_dir': '../DATA/flickr/',
        'input_fn': 'flickr.graph',
        'delimiter': ',',
        'zero_indexed': False
    }

    vog = VoG(subgraph_generation_algo=subgraph_generation_algo,
              hubset_k=hyperparam1,
              gcc_num_nodes_criterion=hyperparam2,
              min_egonet_size=hyperparam1,
              egonet_num_nodes_criterion=hyperparam2,
              hop_k=hyperparam3,
              **kwargs)

    os.system('python ../MDL/score.py ' + normalized_fn + ' ' + str(vog) + ' > ' + str(vog)+'.lgm')

