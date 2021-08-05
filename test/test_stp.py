
from stp import *


def test_stp():
    a = array([[1, 0, 0]]).T
    b = array([[1, 0, 0]]).T
    assert (stp(a, b) == array([[1, 0, 0, 0, 0, 0, 0, 0, 0]]).T).all()
    a = array([[1, 2],
               [3, 4]])
    b = array([[0, 1]]).T
    assert (stp(a, b) == array([[2, 4]]).T).all()


def test_stp_n():
    a = array([[1, 0]]).T
    b = array([[1, 0]]).T
    c = array([[1, 0]]).T
    assert (stp_n(a, b, c) == array([[1, 0, 0, 0, 0, 0, 0, 0]]).T).all()


def test_reducing_matrix():
    arr = array([[1, 0, 0]]).T
    a = stp(arr, arr)
    b = stp(ReducingMatrix(3), arr)
    assert (a == b).all()
    assert id(ReducingMatrix(12)) == id(ReducingMatrix(12))


def test_retriever_matrix():
    a = array([[1, 0, 0, 0]]).T
    b = array([[0, 1, 0, 0]]).T
    c = array([[0, 0, 1, 0]]).T
    d = array([[0, 0, 0, 1]]).T
    e = stp_n(a, b, c, d)
    assert (stp(RetrieverMatrix(4, 4, 1), e) == a).all()
    assert (stp(RetrieverMatrix(4, 4, 2), e) == b).all()
    assert (stp(RetrieverMatrix(4, 4, 3), e) == c).all()
    assert (stp(RetrieverMatrix(4, 4, 4), e) == d).all()

    assert id(RetrieverMatrix(3, 2, 1)) == id(RetrieverMatrix(3, 2, 1))


def test_swap_matrix():
    a = array([[1, 0, 0, 0]]).T
    b = array([[0, 1, 0, 0]]).T

    assert (SwapMatrix(4, 4).stp(stp(a, b)) == stp(b, a)).all()


def test_front_maintaining_operator():
    a = array([[1, 0, 0, 0]]).T
    b = array([[0, 1, 0]]).T

    assert (FrontMaintainingOperator(4, 3).stp(stp(a, b)) == a).all()


def test_rear_maintaining_operator():
    a = array([[1, 0, 0, 0]]).T
    b = array([[0, 1, 0]]).T

    assert (RearMaintainingOperator(4, 3).stp(stp(a, b)) == b).all()


def test_col():
    a = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    assert (col(a, 1) == array([[1, 5, 9]]).T).all()
    assert (col(a, 2) == array([[2, 6, 10]]).T).all()
    assert (col(a, 3) == array([[3, 7, 11]]).T).all()
    assert (col(a, 4) == array([[4, 8, 12]]).T).all()


def test_khatri_rao():
    a = array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    print(khatri_rao(a, a))
