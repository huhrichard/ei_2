import os, sys
from scipy.sparse import coo_matrix, csr_matrix, eye, load_npz, save_npz
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import gzip
import pandas as pd


def normalizeGraphEdgeWeights(W, axis=0):
    """
    *W*: weighted network as a scipy sparse matrix in csr format
    *axis*: The axis to normalize by. 0 is columns, 1 is rows
    """
    # normalize the matrix
    # by dividing every edge weight by the node's degree
    deg = np.asarray(W.sum(axis=axis)).flatten()
    deg = np.divide(1., deg)
    deg[np.isinf(deg)] = 0
    # make sure we're dividing by the right axis
    if axis == 1:
        deg = csr_matrix(deg).T
    else:
        deg = csr_matrix(deg)
    P = W.multiply(deg)
    return P.asformat(W.getformat())


def setup_sparse_network(network_file, node2idx_file=None, forced=False):
    """
    Takes a network file and converts it to a sparse matrix
    """
    sparse_net_file = network_file.replace('.'+network_file.split('.')[-1], '.npz')
    if node2idx_file is None:
        node2idx_file = sparse_net_file + "-node-ids.txt"
    if forced is False and (os.path.isfile(sparse_net_file) and os.path.isfile(node2idx_file)):
        print("Reading network from %s" % (sparse_net_file))
        W = load_npz(sparse_net_file)
        print("\t%d nodes and %d edges" % (W.shape[0], len(W.data)/2))
        print("Reading node names from %s" % (node2idx_file))
        nodes = pd.read_csv(node2idx_file, header=None, index_col=None, squeeze=True).values
    elif os.path.isfile(network_file):
        print("Reading network from %s" % (network_file))
        u,v,w = [], [], []
        open_func = gzip.open if network_file.endswith('.gz') else open
        with open_func(network_file, 'r') as f:
            for line in f:
                line = line.decode() if network_file.endswith('.gz') else line
                if line[0] == '#':
                    continue
                line = line.rstrip().split('\t')
                u.append(line[0])
                v.append(line[1])
                w.append(float(line[2]))
        print("\tconverting uniprot ids to node indexes / ids")
        # first convert the uniprot ids to node indexes / ids
        nodes = sorted(set(list(u)) | set(list(v)))
        node2idx = {prot: i for i, prot in enumerate(nodes)}
        i = [node2idx[n] for n in u]
        j = [node2idx[n] for n in v]
        print("\tcreating sparse matrix")
        #print(i,j,w)
        W = coo_matrix((w, (i, j)), shape=(len(nodes), len(nodes))).tocsr()
        # make sure it is symmetric
        if (W.T != W).nnz == 0:
            pass
        else:
            print("### Matrix not symmetric!")
            W = W + W.T
            print("### Matrix converted to symmetric.")
        #name = os.path.basename(net_file)
        print("\twriting sparse matrix to %s" % (sparse_net_file))
        save_npz(sparse_net_file, W)
        print("\twriting node2idx labels to %s" % (node2idx_file))
        with open(node2idx_file, 'w') as out:
            out.write(''.join(["%s\n" % (n) for n in nodes]))
    else:
        print("Network %s not found. Quitting" % (network_file))
        sys.exit(1)

    return W, nodes


def run_rwr(P, alpha=0.9, eps=1e-4, max_iters=10, verbose=False):
    """
    Run Random Walk with Restarts on a graph
    *P*: noramlized csr scipy sparse matrix
    *alpha*: restart parameter
    *max_iters*: maximum number of iterations
    *eps*: maximum difference of node scores from one iteration to the next
    """

    # intialize with a 1 along each diagonal
    P0 = eye(P.shape[0])
    # matrix of node score vectors
    X = csr_matrix(P.shape)
    prev_X = csr_matrix(P.shape)
    for iters in trange(1,max_iters+1):
        X = alpha*csr_matrix.dot(P, prev_X) + ((1-alpha)*P0)

        max_d = (X - prev_X).max()
        if verbose:
            print("\t\titer %d max score change: %0.6f" % (iters, max_d))
        if max_d < eps:
            # converged!
            break
        prev_X = X.copy()

    if iters == max_iters:
        print("Reached max iters %d" % (max_iters))
    else:
        print("RWR converged after %d iters" % (iters))

    return X


def main(net_file, out_file, **kwargs):

    # load the network
    W, prots = setup_sparse_network(net_file, forced=kwargs.get('forced', False))

    # column-normalize the network
    P = normalizeGraphEdgeWeights(W)
    # run RWR
    X = run_rwr(P, alpha=kwargs['alpha'], eps=kwargs['eps'], max_iters=kwargs['max_iters'], verbose=True)

    # save to a file
    # TODO use a sparse matrix
    A = X.toarray()

    print("Writing %s" % (out_file))
    with open(out_file, 'wb') as out:
        pickle.dump(A, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--results-path', type=str, default='./test_results/', help="Saving results.")
    parser.add_argument('--nets', type=str, nargs='+', help="Network files (edgelist format: i, j, w_ij).")
    parser.add_argument('--alpha', type=float, default=0.9,
            help="RWR restart parameter")
    parser.add_argument('--max-iters', type=int, default=10,
            help="maximum number of iterations to run RWR")
    parser.add_argument('--eps', type=float, default=1e-4,
            help="Maximum difference of node scores from one iteration to the next")
    args = parser.parse_args()
    print (args)
    kwargs = vars(args)

    # make sure the output directory exists
    os.makedirs(os.path.dirname(kwargs['results_path']), exist_ok=True)

    for net_file in args.nets:
        out_file = "%srwr-a%s-maxi%s-eps%s.pckl" % (kwargs['results_path'], kwargs['alpha'], kwargs['max_iters'], kwargs['eps'])
        main(net_file, out_file, **kwargs)