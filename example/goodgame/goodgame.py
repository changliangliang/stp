"""
作者: chang
时间: 2021/3/9
邮箱： changliangliang1996@gmail.com
"""

from stp import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 创建网络
g_1 = nx.Graph()
g_2 = nx.Graph()
g_all = nx.Graph()

g_1.add_nodes_from(["x1", "x2", "x3"])
g_1.add_edges_from([("x1", "x2"), ("x1", "x3"), ("x2", "x3")])

g_2.add_nodes_from(["y1", "y2", "y3"])
g_1.add_edges_from([("y1", "y2"), ("y1", "y3"), ("y2", "y3")])

g_all.add_nodes_from(g_1)
g_all.add_nodes_from(g_2)
g_all.add_edges_from(g_1.edges)
g_all.add_edges_from(g_2.edges)
g_all.add_edges_from([("x1", "y1"), ("x2", "y2"), ("x3", "y3")])
nx.draw(g_all, with_labels=True, node_size=1000)


def N_1_i(i, m, a_i):
    return stp_n([
        np.array([a_i, 0]),
        RearMaintainingOperator(np.power(2, i-1), 2),
        FrontMaintainingOperator(np.power(2, i), np.power(2, m-i))
    ])


def N_2_j(j, n, b_j):
    return stp_n([
        np.array([b_j, 0]),
        RearMaintainingOperator(np.power(2, j-1), 2),
        FrontMaintainingOperator(np.power(2, j), np.power(2, n-j))
    ])


def N_1(a_i_list):
    result = N_1_i(1, len(a_i_list), a_i_list[0])
    for i in range(2, len(a_i_list) + 1):
        result = result + N_1_i(i, len(a_i_list), a_i_list[i - 1])
    return result


def N_2(b_j_list):
    result = N_2_j(1, len(b_j_list), b_j_list[0])
    for i in range(2, len(b_j_list)+1):
        result = result + N_1_i(i, len(b_j_list), b_j_list[i-1])
    return result


def N_1_(a_i_list, m, m_1, n, n_1):
    return stp_n([
        N_1(a_i_list),
        FrontMaintainingOperator(np.power(2, m), np.power(2, n-n_1)),
        FrontMaintainingOperator(np.power(2, m_1), np.power(2, n_1))
    ])


def N_2_(b_j_list, m, m_1, n, n_1):
    return stp_n([
        N_2(b_j_list),
        RearMaintainingOperator(np.power(2, m_1), np.power(2, n)),
        FrontMaintainingOperator(np.power(2, m_1 + n_1), np.power(2, m-m_1)),
    ])


def P_i(N, T1, i, r1, r2, m, n, a_i):

    pa = stp_(
        (r2 * N) / m - N_1_i(i, m, a_i),
        FrontMaintainingOperator(np.power(2, m), np.power(2, n))
    )

    pb = stp_(
        (r1 * N) / m - N_1_i(i, m, a_i),
        FrontMaintainingOperator(np.power(2, m), np.power(2, n))
    )

    N = stp_(
        N,
        FrontMaintainingOperator(np.power(2, m), np.power(2, n))
    )

    result = np.zeros(N.shape)
    for i in range(N.shape[1]):
        if N[0][i] >= T1:
            result[0][i] = pa[0][i]
        else:
            result[0][i] = pb[0][i]
    return result


def C_j(N, T2, j, r1, r2, m, n, b_j):

    pa = stp_(
        (r2 * N) / n - N_2_j(j, n, b_j),
        RearMaintainingOperator(np.power(2, m), np.power(2, n))
    )

    pb = stp_(
        (r1 * N) / n - N_2_j(j, n, b_j),
        RearMaintainingOperator(np.power(2, m), np.power(2, n))
    )

    N = stp_(
        N,
        RearMaintainingOperator(np.power(2, m), np.power(2, n))
    )

    result = np.zeros(N.shape)
    for i in range(N.shape[1]):
        if N[0][i] >= T1:
            result[0][i] = pa[0][i]
        else:
            result[0][i] = pb[0][i]
    return result


a_i_list = [2, 4, 3]
b_j_list = [3, 3, 3]
T1 = 5
T2 = 5
r1 = 2
r2 = 3
m = 3
n = 3

N1 = N_1(a_i_list)
N2 = N_2(b_j_list)

N1_ = N_1_(a_i_list, 3, 1, 3, 1)
N2_ = N_2_(b_j_list, 3, 1, 3, 1)

P1 = P_i(N1, T1, 1, r1, r2, m, n, a_i_list[0])
P2 = P_i(N1, T1, 2, r1, r2, m, n, a_i_list[1])
P3 = P_i(N1, T1, 3, r1, r2, m, n, a_i_list[2])
C1 = C_j(N2, T2, 1, r1, r2, m, n, b_j_list[0])
C2 = C_j(N2, T2, 2, r1, r2, m, n, b_j_list[1])
C3 = C_j(N2, T2, 3, r1, r2, m, n, b_j_list[2])


P = np.zeros((6, 64))
P[0] = P1[0]
P[1] = P2[0]
P[2] = P3[0]
P[3] = C1[0]
P[4] = C2[0]
P[5] = C3[0]










