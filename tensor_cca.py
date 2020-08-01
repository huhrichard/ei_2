"""
Python Implementation of Tensor CCA: https://github.com/yluopku/TCCA

"""
# from sktensor import cp_als, dtensor
# from sktensor import ttm
import tensorly as tl
from tensorly.tenalg import mode_dot, multi_mode_dot
from tensorly.decomposition import parafac
from scipy.linalg import sqrtm
import numpy as np

eps = 2.2204e-16

"""
Tested the whole function
"""
def var_cov_ten_calculation(X):
    nbView = X.shape[0]
    nbSample = X.shape[1]
    var_mats = []
    for v in range(nbView):
        var_mat = np.matmul(X[v].T, X[v])
        var_mat = var_mat/(nbSample-1)
        var_mats.append(var_mat+eps*np.eye(var_mat.shape[0]))

    cov_ten = 0
    for n in range(nbSample):
        u = []
        for v in range(nbView):
            # u.append(tl.tensor(np.expand_dims(X[v, n, :].T, axis=1)))
            u.append(tl.tensor(np.expand_dims(X[v][n, :], axis=1)))
            # print(u[-1])
        # print()
        # u = np.array(u)
        # cov_x = ktensor(u)
        # cov_x = tl.kruskal_to_tensor((np.ones((X.shape[-1])), u))
        # print(u)
        cov_x = tl.kruskal_to_tensor((np.ones((u[0].shape[1])), u))
        # print(cov_x.shape)
        # print(cov_x.shape)
        cov_ten += cov_x

    cov_ten = cov_ten/(nbSample-1)

    return var_mats, cov_ten


def tcca(X, var_mats, cov_ten, **kargs):
    use_inv2 = kargs.get('use_inv2', True)
    rDim = kargs.get('rDim', 3)
    # eps = kargs.get('eps', 1e-6)
    var_mats_inv2 = []
    nbV = X.shape[0]

    for v in range(nbV):
        # var_mat_inv2 = (var_mats[v] + eps * np.eye(len(var_mats[v])))**(-0.5)
        var_mat_plus_eps = var_mats[v] + eps * np.eye(len(var_mats[v]))
        """
        Not matched with matlab
        """
        var_mat_inv2 = sqrtm(np.linalg.inv((var_mat_plus_eps)))
        # print(np.matmul(np.matmul(var_mat_inv2, var_mat_inv2), var_mat_plus_eps))
        var_mats_inv2.append(var_mat_inv2)

    # print(var_mats_inv2[0]/(var_mats_inv2[0][0,0]))
    var_mats_inv2 = np.array(var_mats_inv2)
    # Tensor times matrix (MATLAB: ttm)
    # M_ten = mode_dot(cov_ten, var_mats_inv2)
    M_ten = multi_mode_dot(cov_ten, var_mats_inv2)
    # print(M_ten[0,0])
    # print(M_ten[0,1])
    # print(M_ten[1,0])
    # print('M_ten', M_ten[0,0]/M_ten[0,0,0,0])
    # cp_als???
    """
    tested
    """
    P_kten = parafac(M_ten, rDim, normalize_factors=True)
    # print(P_kten[1])
    Z = []
    H = []
    for v in range(nbV):
        u = P_kten[1]
        # print(u)
        if use_inv2:
            H.append(np.matmul(var_mats_inv2[v], u[v]))
        else:
            H.append(u[v])

        Z.append(np.matmul(X[v],H[v]))
    H = np.array(H)
    Z = np.array(Z)
    return H, Z

# test_tensor = np.random.rand(4,3,2)

if __name__ == "__main__":
    # test_tensor = np.array([[
    #     [0.825506858446406, 0.524778652249267, 0.311750614229474, 0.672925782237633, 0.685311331732809, 0.465699416802125],
    #                        [0.293769908740078, 0.954808353425380, 0.592573354762315, 0.863622067898336, 0.0275524333247693, 0.0100981933111782],
    #                        [0.390856194491897, 0.463996020361779, 0.380373490220110, 0.896095167331442, 0.0737468893983154, 0.720334980972267],
    #                        [0.120060825310676, 0.381114220424143, 0.937250129383640, 0.671477339543159, 0.463970117486999, 0.670025200895046],
    #                        [0.862290996528366, 0.386702668567515, 0.362924550828955, 0.519645701870157, 0.945764603999334, 0.113551443029918]],
    #
    # [[0.459103003823712, 0.464533049829552, 0.590125923319778, 0.694653678077578, 0.345901608628175, 0.332860111879585],
    # [0.144185265127656, 0.406968485287842, 0.246393235515055, 0.0443674498444272, 0.568336213737883, 0.379900360191195],
    # [0.321751000509481, 0.919332232754367, 0.131109301527547, 0.683729535287120, 0.0267442925957703, 0.985353733406101],
    # [0.669065562854086, 0.216835902639123, 0.554030773487888, 0.442299570628581, 0.491325657112210, 0.840197356343361],
    # [0.984914888620803, 0.361015655734118, 0.595210010928665, 0.728883554567063, 0.710319084911756, 0.196968062872518]],
    #
    # [[0.442232698408064,0.386662607726950,0.397489981910908,0.889807912469752,0.641301723495732,0.675857721211323],
    # [0.604962144328729,0.638448066841088,0.614125976624957,0.00158487912341248,0.358066792766595,0.140720398696475],
    # [0.598759732953627,0.396659398170495,0.615289135656920,0.637063427193586,0.0285860281570750,0.0877408916539237],
    # [0.354831751601227,0.654505051511575,0.164502985852073,0.125020842591776,0.445375235958665,0.661166027496181],
    # [0.657249161813990,0.266400755041466,0.801683215797731,0.225301439978788,0.421590757286917,0.217424128305830]],
    #
    # [[0.247324436086847, 0.934175586297259, 0.470679132487010, 0.643041028940604, 0.567333036832201, 0.723739899566276],
    # [0.781797311632800, 0.689078898024316, 0.704179709173823, 0.590573404724977, 0.166350038799620, 0.896492010399928],
    # [0.667416612462800, 0.349039094685129, 0.148372084437111, 0.767213212683431, 0.845102105217568, 0.341218791595296],
    # [0.271005662284328, 0.300064372969821, 0.689623134531837, 0.552465570416518, 0.631082391660233, 0.864230342146117],
    # [0.792166012708175, 0.181916503786071, 0.496022076576958, 0.288841990422850, 0.601756633539571, 0.119360449779481]]]
    #                        )
    print('test')
    test_tensor = [np.random.randn(20000, 10) for n in range(5)]
    print(test_tensor.shape)

    var_mats, cov_t = var_cov_ten_calculation(test_tensor)
    # print(cov_t[0,0])
    H, Z = tcca(test_tensor, var_mats=var_mats, cov_ten=cov_t)
    # print(H.shape, Z.shape)
    print('H:\n', H[0])
    print('Z:\n', Z[0])
    # print(cov_t.shape)

    # a = np.array([[[0.6891, 0.1819, 0.1484],
    #               [0.3490, 0.4707, 0.6896],
    #               [0.3001, 0.7042, 0.4960]],
    #               [[0.6430, 0.5525, 0.1664],
    #                [0.5906, 0.2888, 0.8451],
    #                [0.7672, 0.5673, 0.6311]],
    #               [[0.6018, 0.3412, 0.0911],
    #                [0.7237, 0.8642, 0.7722],
    #                [0.8965, 0.1194, 0.5480]]]
    #              )
    # a = a.transpose(1,2,0)
    # # P = parafac(a, 2, normalize_factors=True)
    # # print(P[0],P[1])
    #
    # a_dt = dtensor(a)
    # # M = a_dt.ttm([a[i] for i in range(a.shape[0])])
    # M = multi_mode_dot(a, a)
    # print(M)

