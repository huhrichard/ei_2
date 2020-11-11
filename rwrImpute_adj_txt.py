from rwr_from_jeff import *
import csv

edge_txt_format = 'human_string_{}_adjacency.txt'

sub_network_list = ['coexpression', 'cooccurence', 'database', 'experimental', 'fusion', 'neighborhood']
txt_dir = 'not_on_github/edge_list/'
node_fn = txt_dir + 'human_string_genes.txt'
csv_dir = 'not_on_github/csv/'
term_dir = 'not_on_github/tsv/'
term_fn = 'GO2HPO_binary.tsv'


def setup_sparse_net_v2(network_file, node2idx_file=node_fn):
    sparse_net_file = network_file.replace('.' + network_file.split('.')[-1], '.npz')
    nodes = pd.read_csv(node2idx_file, header=None, index_col=None, squeeze=True).values
    u, v, w = np.loadtxt(network_file).T
    u = (u-1).astype(int)
    v = (v-1).astype(int)
    w[w<0] = 0
    print('# of nodes:', len(u))

    print('# of 0', np.sum(w == 0))

    # node2idx = {prot: i for i, prot in enumerate(nodes)}
    # i = [node2idx[n] for n in u]
    # j = [node2idx[n] for n in v]
    W = coo_matrix((w, (u, v)), shape=(len(nodes), len(nodes))).tocsr()
    # make sure it is symmetric
    print(W.shape)
    if (W.T != W).nnz == 0:
        pass
    else:
        print(network_file)
        print("### Matrix not symmetric!")
        W = W + W.T
        print("### Matrix converted to symmetric.")


    # save_npz(sparse_net_file, W)

    return W, nodes, (w == 0)

def main_v2(net_file, out_file, node_file=node_fn, **kwargs):
    W, prots, original_0 = setup_sparse_net_v2(net_file, node2idx_file=node_file)
    print(prots)
    # net_df = pd.DataFrame(data=W, index=list(prots), columns=list(prots))
    # net_df = pd.DataFrame(data=W)

    print(np.sum(original_0))
    W = W.toarray()
    # print((W==0).all(axis=1))
    W_0_bool = ~(W != 0)
    # net_df = net_df.loc[~((net_df==0).all(axis=1))]
    print(W.shape)
    W_keep = ~(np.all((W == 0), axis=1))

    # print(W_keep.shape)
    # print(np.sum(W_keep))
    W_filtered_0 = W[W_keep, :]
    W_0_bool_filtered = W_0_bool[W_keep, :]
    zero_count = np.sum(W_filtered_0 == 0)
    # net_df.drop(how='all', inplace=True)
    # net_df.drop(how='all', inplace=True)
    # print('missing value count', net_file, zero_count)
    # print('missing value %', net_file, zero_count/(W_filtered_0.shape[0]*W_filtered_0.shape[1]))

    return W, W_keep

    # go_term_df = pd.read_csv(term_dir+term_fn, sep='\t', index_col=0)


    # column-normalize the network
    # P = normalizeGraphEdgeWeights(W)
    # run RWR
    # X = run_rwr(P, verbose=True)

    # A=X.toarray()

    # imputed_network_edge_fn = net_file.replace('.'+net_file.split('.')[-1], '_rwrImputed.txt')
    # with open(imputed_network_edge_fn, 'w') as f:
    #     writer = csv.writer(f)
    #     for (n, m), val in np.ndenumerate(A):
    #         if val != 0:
    #             writer.writerow([int(n+1), int(m+1), val])

    # filled_df = pd.DataFrame(A, index=prots, columns=prots)
    # print(A.shape, filled_df.shape)

    # filled_df.to_csv(out_file, index_label=False)

    # edge_list

    # save to a file
    # TODO use a sparse matrix
    # A = X.toarray()
    #
    #
    # print("Writing %s" % (out_file))
    # with open(out_file, 'w') as out:
    #     pickle.dump(A, out)

W_list = []
W_keep_list = []
for sub_net in sub_network_list:
    W, W_keep = main_v2(net_file=txt_dir+edge_txt_format.format(sub_net),
            out_file="{}rwrImputed_{}.csv".format(csv_dir, sub_net))
    W_list.append(W)
    W_keep_list.append(W_keep)

W_keep_bool = np.ones_like(W_keep_list[0], dtype=bool)
for idx, sub_net in enumerate(sub_network_list):
    W_keep_bool = W_keep_bool * W_keep_list[idx]

for idx, sub_net in enumerate(sub_network_list):
    W_0_filtered = W_list[idx][W_keep_bool,:]
    zero_count = np.sum(W_0_filtered == 0)
    print('missing value count', sub_net, zero_count)
    print('missing value of {} = {} %'.format(sub_net,
                                              100*zero_count / (W_0_filtered.shape[0] * W_0_filtered.shape[1])))

