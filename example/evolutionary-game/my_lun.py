import networkx as nx
import numpy as np
from stp import *


class Fai(np.ndarray):

    def __new__(cls, k, tao):
        result_matrix = RetrieverMatrix(tao, k, 1)
        result_matrix.fill(0)
        for l in range(1, tao + 1):
            result_matrix = result_matrix + RetrieverMatrix(tao, k, l)
        return np.divide(result_matrix, tao).view(cls)


class C(np.ndarray):

    def __new__(cls, n, k, tao):
        fai = Fai(k, tao)
        descending_power_matrix = DescendingPowerMatrix(
            n - 1, np.power(k, tao))
        result_matrix = stp_(fai, RetrieverMatrix(n - 1, np.power(k, tao), 1))
        identity = np.identity(np.power(k, (n - 1) * tao))
        for j in range(2, n):
            process_a = stp_(fai, RetrieverMatrix(n - 1, np.power(k, tao), j))
            process_b = np.kron(identity, process_a)
            process_c = stp_(process_b, descending_power_matrix)
            result_matrix = stp_(result_matrix, process_c)
        return result_matrix.view(cls)


# 网络
graph = nx.Graph()
graph.add_nodes_from([1, 2, 3])
graph.add_edges_from([(1, 2), (1, 3), (2, 3)])


# 网络基础博弈的结构矩阵
M_c = np.array([2, 0, 1, 3], dtype=int)


# 计算Mi
def M_i(i, k, n):
    d = FrontMaintainingOperator(k, np.power(k, n-2))
    process_a = stp_(M_c, SwapMatrix(k, k))
    process_b = None
    first_flag = True
    for j in list(graph.adj[i]):
        if first_flag:
            if j < i:
                process_b = stp_(d, SwapMatrix(np.power(k, j - 1), k))
            elif j > i:
                process_b = stp_(d, SwapMatrix(np.power(k, j - 2), k))
            first_flag = False
        else:
            if j < i:
                process_b = process_b + stp_(d, SwapMatrix(np.power(k, j - 1), k))

            elif j > i:
                process_b = process_b + \
                            stp_(d, SwapMatrix(np.power(k, j - 2), k))
    return stp_(process_a, process_b)


# 计算M_i_c

def M_i_c(i, k, n, tao):
    return stp_(M_i(i, k, n), C(n, k, tao))


def L_i_bo_lang(i, k, n, tao):
    m_i_c = M_i_c(i, k, n, tao)
    # ξ
    m_blocks = np.hsplit(m_i_c, np.power(k, (n - 1)*tao))
    xi_list = []
    for m_block in m_blocks:
        xi_list.append(np.where(m_block == np.max(m_block))[1][0])

    l = LogicMatrix(2, list_=xi_list)
    return l


def L_i_x(i, k, n, tao):
    return stp_(stp_(L_i_bo_lang(i, k, n, tao), RearMaintainingOperator(np.power(k, tao), np.power(k, (n - 1) * tao))),
                SwapMatrix(np.power(k, (i - 1) * tao), np.power(k, tao)))


def L_i(i, k, n, tao):
    process_a = stp_(RearMaintainingOperator(k, k),
                     FrontMaintainingOperator(np.power(k, tao), np.power(k, (n - 1) * tao)))

    process_b = np.kron(np.identity(np.power(k, n * tao), dtype=int),
                        L_i_x(i, k, n, tao))

    process_c = stp_(process_a,
                     SwapMatrix(np.power(k, (i - 1) * tao), np.power(k, tao)))

    process_d = stp_(process_c, process_b)
    return stp_(process_d, DescendingPowerMatrix(n, np.power(k, tao)))


def khatri_rao(a, b):

    if len(a.shape) == 1:
        a.shape = (1, a.shape[0])
    if len(b.shape) == 1:
        b.shape = (1, b.shape[0])
    q, s = a.shape
    p, s = b.shape
    result_matrix = np.zeros((p*q, s))
    for i in range(s):
        result_matrix[:, i] = np.kron(a[:, i], b[:, i])
    return result_matrix


def L(k, n, tao):
    l = L_i(1, k, n, tao)
    for j in range(2, n + 1):
        l = khatri_rao(l, L_i(j, k, n, tao))
    return l


def L_v(k, n, tao):
    l = L_i(2, k, n, tao)
    for j in range(3, n + 1):
        l = khatri_rao(l, L_i(j, k, n, tao))
    return l


l = L_v(2, 3, 2)
print(type(l))
l = LogicMatrix.to_logic_matrix(l)
