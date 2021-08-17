import numpy as np

from stp import *


def test_stp_matrix():
    """
    StpMatrix矩阵测试
    """
    matrix = StpMatrix((2, 1), dtype=int, buffer=np.array([[2], [1]]))
    assert matrix == np.array([[2], [1]])


def test_stp():
    a = array([[1, 0, 0]]).T
    b = array([[1, 0, 0]]).T
    assert (stp(a, b) == array([[1, 0, 0, 0, 0, 0, 0, 0, 0]]).T)
    a = array([[1, 2],
               [3, 4]])
    b = array([[0, 1]]).T
    assert (stp(a, b) == array([[2, 4]]).T)


def test_col():
    matrix = array([[1, 3], [2, 4]])
    assert col(matrix, 1) == array([[1], [2]])


def test_stp_n():
    a = array([[1, 0]]).T
    b = array([[1, 0]]).T
    c = array([[1, 0]]).T
    assert (stp_n(a, b, c) == array([[1, 0, 0, 0, 0, 0, 0, 0]]).T)


def test_reducing_matrix():
    arr = array([[1, 0, 0]]).T
    a = stp(arr, arr)
    b = stp(ReducingMatrix(3), arr)
    assert (a == b)
    assert id(ReducingMatrix(12)) == id(ReducingMatrix(12))


def test_retriever_matrix():
    a = array([[1, 0, 0, 0]]).T
    b = array([[0, 1, 0, 0]]).T
    c = array([[0, 0, 1, 0]]).T
    d = array([[0, 0, 0, 1]]).T
    e = stp_n(a, b, c, d)
    assert (stp(RetrieverMatrix(4, 4, 1), e) == a)
    assert (stp(RetrieverMatrix(4, 4, 2), e) == b)
    assert (stp(RetrieverMatrix(4, 4, 3), e) == c)
    assert (stp(RetrieverMatrix(4, 4, 4), e) == d)

    assert id(RetrieverMatrix(3, 2, 1)) == id(RetrieverMatrix(3, 2, 1))


def test_swap_matrix():
    a = array([[1, 0, 0, 0]]).T
    b = array([[0, 1, 0, 0]]).T

    assert (SwapMatrix(4, 4).stp(stp(a, b)) == stp(b, a))


def test_front_maintaining_operator():
    a = array([[1, 0, 0, 0]]).T
    b = array([[0, 1, 0]]).T

    assert (FrontMaintainingOperator(4, 3).stp(stp(a, b)) == a)


def test_rear_maintaining_operator():
    a = array([[1, 0, 0, 0]]).T
    b = array([[0, 1, 0]]).T

    assert (RearMaintainingOperator(4, 3).stp(stp(a, b)) == b)


def test_khatri_rao():
    a = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    print(khatri_rao(a, a))


def test_logic_matrix():
    a = array([[1, 1, 1, 1], [0, 0, 0, 0]])
    b = a.view(LogicMatrix)
    print(b)
