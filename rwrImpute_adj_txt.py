from rwr_from_jeff import *
import csv

edge_txt_format = 'human_string_{}_adjacency.txt'

sub_network_list = ['coexpression', 'cooccurence', 'database', 'experimental', 'fusion', 'neighborhood']
txt_dir = 'not_on_github/edge_list/'
node_fn = txt_dir + 'human_string_genes.txt'
csv_dir = 'not_on_github/csv/'

def setup_sparse_net_v2(network_file, node2idx_file=node_fn):
    sparse_net_file = network_file.replace('.' + network_file.split('.')[-1], '.npz')
    nodes = pd.read_csv(node2idx_file, header=None, index_col=None, squeeze=True).values
    u, v, w = np.loadtxt(network_file).T
    u = (u-1).astype(int)
    v = (v-1).astype(int)
    w[w<0] = 0
    print('# of nodes:', len(u))

    # node2idx = {prot: i for i, prot in enumerate(nodes)}
    # i = [node2idx[n] for n in u]
    # j = [node2idx[n] for n in v]
    W = coo_matrix((w, (u, v)), shape=(len(nodes), len(nodes))).tocsr()
    # make sure it is symmetric
    if (W.T != W).nnz == 0:
        pass
    else:
        print(network_file)
        print("### Matrix not symmetric!")
        W = W + W.T
        print("### Matrix converted to symmetric.")


    # save_npz(sparse_net_file, W)

    return W, nodes

def main_v2(net_file, out_file, node_file=node_fn, **kwargs):
    W, prots = setup_sparse_net_v2(net_file, node2idx_file=node_file)

    # column-normalize the network
    P = normalizeGraphEdgeWeights(W)
    # run RWR
    X = run_rwr(P, verbose=True)

    A=X.toarray()

    imputed_network_edge_fn = net_file.replace('.'+net_file.split('.')[-1], '_rwrImputed.txt')
    with open(imputed_network_edge_fn, 'w') as f:
        writer = csv.writer(f)
        for (n, m), val in np.ndenumerate(A):
            if val != 0:
            writer.writerow([int(n+1), int(m+1), val])

    filled_df = pd.DataFrame(A, index=prots, columns=prots)
    print(A.shape, filled_df.shape)

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
    main_v2(net_file=txt_dir+edge_txt_format.format(sub_net),
            out_file="{}rwrImputed_{}.csv".format(csv_dir, sub_net))

