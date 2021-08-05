"""
半张量积计算相关函数

作者: chang
时间: 2021/1/26
邮箱： changliangliang1996@gmail.com
"""

import numpy as np
from .error import StpTypeError


class StpMatrix(np.ndarray):

    """
    该包下所有矩阵的父类，提供 stp 方法, 继承自 np.ndarray
    """
    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        matrix = super().__new__(cls, shape, dtype, buffer, offset, strides, order)
        return matrix.view(StpMatrix)

    def stp(self, matrix) -> 'StpMatrix':
        """
        求本矩阵和其他矩阵的半张量积

        Parameters
        ----------
        matrix : StpMatrix
            矩阵

        Returns
        -------
        StpMatrix
        """
        if not isinstance(matrix, StpMatrix):
            raise StpTypeError("接受的参数必须为 StpMatrix类型")
        return stp(self, matrix)

    def khatri_rao(self, matrix) -> 'StpMatrix':
        if not isinstance(matrix, StpMatrix):
            raise StpTypeError("接受的参数必须为 StpMatrix类型")
        return khatri_rao(self, matrix)


def col(matrix: StpMatrix, i):
    m, n = matrix.shape
    return matrix[:, i - 1].reshape(m, 1).view(StpMatrix)


def khatri_rao(matrix_a: StpMatrix, matrix_b: StpMatrix):

    q, s = matrix_a.shape
    p, s = matrix_b.shape

    matrix = zeros((p * q, s), dtype=int)
    for i in range(s):
        matrix[:, i] = stp(col(matrix_a, i + 1), col(matrix_b, i + 1))[:, 0]

    return matrix.view(StpMatrix)


def stp(matrix_a: StpMatrix, matrix_b: StpMatrix) -> StpMatrix:
    """
    计算两个矩阵的半张量积

    Parameters
    ----------
    matrix_a : StpMatrix
        第一个矩阵
    matrix_b : StpMatrix
        第二个矩阵

    Returns
    -------
    StpMatrix
        matrix_a 和 matrix_b 做半张量积运算后结果。
    """

    if not (isinstance(matrix_a, StpMatrix) and isinstance(matrix_b, StpMatrix)):
        raise StpTypeError("接受的参数必须为StpMatrix类型")

    if len(matrix_a.shape) == 1:
        raise StpTypeError("参数必须为二维矩阵")
    if len(matrix_b.shape) == 1:
        raise StpTypeError("参数必须为二维矩阵")

    m, n = matrix_a.shape
    p, t = matrix_b.shape
    alpha = np.lcm(n, p)

    a = np.kron(matrix_a, np.identity(
        np.divide(alpha, n).astype(int), dtype=int))

    b = np.kron(matrix_b, np.identity(
        np.divide(alpha, p).astype(int), dtype=int))

    return np.dot(a, b).view(type(matrix_b))


def stp_n(*matrix: StpMatrix) -> StpMatrix:
    """
    计算多个矩阵的半张量积

    Parameters
    ----------
    matrix : StpMatrix

    Returns
    -------
    StpMatrix
        多个矩阵做半张量积后的结果
    """
    if len(matrix) <= 1:
        raise StpTypeError("stp_n 参数个数必须大于2")
    result = array([[1]])
    for m in matrix:
        result = stp(result, m)
    return result


def array(p_object, dtype=int, *args, **kwargs) -> StpMatrix:
    """
    创建一个矩阵，使用方式和 np.array一样。
    """
    return np.array(p_object, dtype, *args, **kwargs).view(StpMatrix)


def zeros(shape, dtype=None, order='C', *args, **kwargs):
    return np.zeros(shape, dtype, order, *args, **kwargs).view(StpMatrix)


class RetrieverMatrix(StpMatrix):
    """
    还原矩阵

    Parameters
    ----------
    number : int
        逻辑变量个数
    dimension : int
        逻辑变量维数
    index : int
        要还原的逻辑变量位置
    """

    matrix_cache = {}

    def __new__(cls, number: int, dimension: int, index: int):

        if (number, dimension, index) not in cls.matrix_cache:
            one_s_t_1 = np.empty(np.power(dimension, index - 1), dtype=int)
            one_s_t_2 = np.empty(np.power(dimension, number - index), dtype=int)
            one_s_t_1.fill(1)
            one_s_t_2.fill(1)
            matrix = np.kron(one_s_t_1, np.kron(np.identity(dimension, dtype=int), one_s_t_2))

            cls.matrix_cache[(number, dimension, index)] = matrix.view(RetrieverMatrix)
        return cls.matrix_cache[(number, dimension, index)]


class SwapMatrix(StpMatrix):
    """
    交换矩阵

    Parameters
    ----------
    dimension_a : int
        第一个逻辑变量维数
    dimension_b : int
        第二个逻辑变量维数
    """

    matrix_cache = {}

    def __new__(cls, dimension_a, dimension_b):

        if (dimension_a, dimension_b) not in cls.matrix_cache:
            matrix = np.zeros((dimension_a * dimension_b, dimension_a * dimension_b), dtype=int)
            for i in range(dimension_a):
                for j in range(dimension_b):
                    col = i * dimension_b + j
                    row = j * dimension_a + i
                    matrix[row][col] = 1
            cls.matrix_cache[(dimension_a, dimension_b)] = matrix.view(cls)
        return cls.matrix_cache[(dimension_a, dimension_b)]


class FrontMaintainingOperator(StpMatrix):
    """
    前保留算子

    Parameters
    ----------
    dimension_a : int
        第一个逻辑变量维数
    dimension_b : int
        第二个逻辑变量维数
    """

    matrix_cache = {}

    def __new__(cls, dimension_a, dimension_b):

        if (dimension_a, dimension_b) not in cls.matrix_cache:
            matrix = np.zeros((dimension_a, dimension_a * dimension_b), dtype=int)
            for i in range(dimension_a):
                for j in range(dimension_b):
                    matrix[i][i * dimension_b + j] = 1
            cls.matrix_cache[(dimension_a, dimension_b)] = matrix.view(cls)
        return cls.matrix_cache[(dimension_a, dimension_b)]


class RearMaintainingOperator(StpMatrix):
    """
    后保留算子

    Parameters
    ----------
    dimension_a : int
        第一个逻辑变量维数
    dimension_b : int
        第二个逻辑变量维数
    """

    matrix_cache = {}

    def __new__(cls, dimension_a, dimension_b):

        if (dimension_a, dimension_b) not in cls.matrix_cache:
            matrix = np.zeros((dimension_b, dimension_a * dimension_b), dtype=int)
            for i in range(dimension_a):
                for j in range(dimension_b):
                    matrix[j][i * dimension_b + j] = 1
            cls.matrix_cache[(dimension_a, dimension_b)] = matrix.view(cls)
        return cls.matrix_cache[(dimension_a, dimension_b)]


class ReducingMatrix(StpMatrix):
    """
    基础降幂矩阵

    Parameters
    ----------
    dimension : int
        要进行降幂的逻辑向量的维数
    """

    matrix_cache = {}

    def __new__(cls, dimension):
        if dimension not in cls.matrix_cache:
            matrix = np.zeros((dimension * dimension, dimension), dtype=int)
            for i in range(dimension):
                matrix[i * dimension + i][i] = 1
            cls.matrix_cache[dimension] = matrix.view(ReducingMatrix)
        return cls.matrix_cache[dimension]


class LogicMatrix(StpMatrix):
    """
    逻辑矩阵

    Parameters
    ----------
    m : int
        矩阵行数
    n : int
        矩阵列数
    values : list[int]
        每列对应的变量值
    """

    def __new__(cls, m, n, values: list[int]):

        matrix = np.ndarray.__new__(cls, (m, n), dtype=int)
        matrix.fill(0)
        for i in range(n):
            matrix[values[i] - 1][i] = 1
        matrix.values = values
        return matrix

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.values = getattr(obj, 'values', None)

    def __str__(self):
        return "LogicMatrix" + str(self.values)

    @staticmethod
    def as_logic_matrix(matrix: StpMatrix):
        m, n = matrix.shape
        values = []
        for i in range(n):
            values.append(matrix[:, i].nonzero()[0][0] + 1)
        matrix.values = values

        return matrix.view(LogicMatrix)


class LogicValue(np.ndarray):
    """
    逻辑变量

    Parameters
    ----------
    n : int
        矩阵行数
    value : int
        该逻辑变量对应的数值
    """
    def __new__(cls, n, value):
        matrix = np.ndarray.__new__(cls, (n, 1), dtype=int)
        matrix.fill(0)
        matrix[value - 1][0] = 1
        setattr(matrix, "value", value)
        return matrix.view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.value = getattr(obj, 'value', None)

    def __str__(self):
        return "LogicValue[" + str(self.value) + "]"
