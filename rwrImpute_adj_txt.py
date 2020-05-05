from rwr_from_jeff import *


edge_txt_format = 'human_string_{}_adjacency.txt'

sub_network_list = ['coexpression', 'cooccurence', 'database', 'experimental', 'fusion', 'neighborhood']
txt_dir = 'not_on_github/edge_list/'
node_fn = txt_dir + 'human_string_genes.txt'
csv_dir = 'not_on_github/csv/'

def setup_sparse_net_v2(network_file, node2idx_file=node_fn):
    sparse_net_file = network_file.replace('.' + network_file.split('.')[-1], '.npz')
    nodes = pd.read_csv(node2idx_file, header=None, index_col=None, squeeze=True).values
    u, v, w = [], [], []
    with open(network_file, 'r') as f:
        for line in f:
            line = line.decode() if network_file.endswith('.gz') else line
            if line[0] == '#':
                continue
            line = line.rstrip().split('\t')
            u.append(line[0])
            v.append(line[1])
            w.append(float(line[2]))
    print(min(u), min(v))

    node2idx = {prot: i for i, prot in enumerate(nodes)}
    i = [node2idx[n] for n in u]
    j = [node2idx[n] for n in v]
    W = coo_matrix((w, (i, j)), shape=(len(nodes), len(nodes))).tocsr()
    # make sure it is symmetric
    if (W.T != W).nnz == 0:
        pass
    else:
        print("### Matrix not symmetric!")
        W = W + W.T
        print("### Matrix converted to symmetric.")
    save_npz(sparse_net_file, W)

    return W, nodes

def main_v2(net_file, out_file, node_file=node_fn, **kwargs):
    W, prots = setup_sparse_network(net_file, node2idx_file=node_file, forced=kwargs.get('forced', False))

    # column-normalize the network
    P = normalizeGraphEdgeWeights(W)
    # run RWR
    X = run_rwr(P, alpha=kwargs['alpha'], eps=kwargs['eps'], max_iters=kwargs['max_iters'], verbose=True)

    filled_df = pd.DataFrame(X.toarray(), index=prots, columns=prots)

    filled_df.to_csv(out_file, index_label=False)
    # edge_list

    # save to a file
    # TODO use a sparse matrix
    # A = X.toarray()
    #
    #
    # print("Writing %s" % (out_file))
    # with open(out_file, 'w') as out:
    #     pickle.dump(A, out)


for sub_net in sub_network_list:
    main_v2(net_file=edge_txt_format.format(sub_net),
            out_file="{}rwrImputed_{}.csv".format(csv_dir, sub_net))

