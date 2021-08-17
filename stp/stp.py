"""
半张量积相关的函数和矩阵

作者: chang
时间: 2021/1/26
邮箱： changliangliang1996@gmail.com
"""

import numpy as np
from .error import StpTypeError


class StpMatrix(np.ndarray):
    """
    StpMatrix(shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None)

        该模块下所有矩阵的父类，继承自 np.ndarray，提供了诸如 stp 之类的公共方法。
        构造函数中的参数以及含义均和 np.ndarray 中类似。
    """

    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        matrix = super().__new__(cls, shape, dtype, buffer, offset, strides, order)
        return matrix.view(StpMatrix)

    def stp(self, matrix) -> 'StpMatrix':
        """
        m.stp(matrix)
            返回矩阵 m 和 matrix 的半张量积

            Parameters
            ----------
            matrix : StpMatrix
                类型为 StpMatrix 的矩阵。

            Returns
            -------
            StpMatrix
                两个矩阵的半张量积
        """
        if not isinstance(matrix, StpMatrix):
            raise StpTypeError("接受的参数必须为 StpMatrix类型")
        return stp(self, matrix)

    def khatri_rao(self, matrix) -> 'StpMatrix':
        """
        m.khatri_rao(matrix)
            计算矩阵 m 和 matrix 的 khatri_rao 积

            Parameters
            ----------
            matrix : StpMatrix
                类型为 StpMatrix 的矩阵

            Returns
            -------
            StpMatrix
                两个矩阵的 khatri_rao 积
        """
        if not isinstance(matrix, StpMatrix):
            raise StpTypeError("接受的参数必须为 StpMatrix类型")
        return khatri_rao(self, matrix)

    def col(self, column: int):
        """
        m.col(column)
            获取矩阵 m 第 column 列

            Parameters
            ----------
            column : int
        """

        return col(self, column=column)

    def __str__(self):
        """
        为了打印美观，重写了 __str__ 方法
        """
        return "StpMatrix" + super().__str__().replace(" [", "          [")

    def __eq__(self, other):
        """
        m.__eq__(other)
            为了简化后面的比较，重写了 __eq__ 方法，当 other 的类型为 np.ndarray 或其子类时，会逐个比较
            m 和 other 中的元素是否相同，如果全部相同则返回 True, 否则返回 False；当 other 为其他类型时，
            直接返回运算 `m is other` 的结果.
        """
        if not isinstance(other, np.ndarray):
            return self is other
        if super().__eq__(other).all():
            return True
        return False

    def as_logic(self):
        return LogicMatrix.of(self)


def stp(matrix_a: 'StpMatrix', matrix_b: 'StpMatrix') -> 'StpMatrix':
    """
    stp(matrix_a, matrix_b)

        计算两个矩阵 matrix_a 和 matrix_b 的半张量积

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

    # 如果参数不是StpMatrix类型则抛出错误
    if not (isinstance(matrix_a, StpMatrix) and isinstance(matrix_b, StpMatrix)):
        raise StpTypeError("接受的参数必须为StpMatrix类型")

    # 如果参数为行向量, 则转化为 1 * n 的矩阵
    if len(matrix_a.shape) == 1:
        matrix_a.reshape((1, matrix_a.shape[0]))
    if len(matrix_b.shape) == 1:
        matrix_b.reshape((1, matrix_b.shape[0]))

    # 根据半张量积的公式计算两个矩阵的半张量积
    m, n = matrix_a.shape
    p, t = matrix_b.shape
    alpha = np.lcm(n, p)
    a = np.kron(matrix_a, np.identity(
        np.divide(alpha, n).astype(int), dtype=int))

    b = np.kron(matrix_b, np.identity(
        np.divide(alpha, p).astype(int), dtype=int))
    return np.dot(a, b).view(StpMatrix)


def khatri_rao(matrix_a: StpMatrix, matrix_b: StpMatrix):
    """
    khatri_rao(matrix_a, matrix_b)

        计算矩阵 matrix_a 和 matrix_b 的 khatri_rao 积。

        Parameters
        ----------
        matrix_a : StpMatrix
            第一个矩阵
        matrix_b : StpMatrix
            第二个矩阵

        Returns
        -------
        StpMatrix
            矩阵 matrix_a 和 matrix_b 的 khatri_rao 积。
    """

    # 如果参数不是StpMatrix类型则抛出错误
    if not (isinstance(matrix_a, StpMatrix) and isinstance(matrix_b, StpMatrix)):
        raise StpTypeError("接受的参数必须为StpMatrix类型")

    # 如果参数为行向量, 则转化为 1 * n 的矩阵
    if len(matrix_a.shape) == 1:
        matrix_a.reshape((1, matrix_a.shape[0]))
    if len(matrix_b.shape) == 1:
        matrix_b.reshape((1, matrix_b.shape[0]))

    # 判断矩阵列数是否相同
    q, s = matrix_a.shape
    p, t = matrix_b.shape
    if s != t:
        raise StpTypeError("两个矩阵的列数不相同")

    # khatri_rao 计算公式
    matrix = zeros((p * q, s), dtype=int)
    for i in range(s):
        matrix[:, i] = stp(col(matrix_a, i + 1), col(matrix_b, i + 1))[:, 0]
    return matrix.view(StpMatrix)


def col(matrix: StpMatrix, column: int):
    """
    col(matrix, column)

        获取矩阵第 column 列

        Parameters
        ----------
        matrix : StpMatrix
            类型为StpMatrix的二维矩阵
        column : int
            大于 0 小于等于 matrix 列数的整数

        Returns
        -------
        StpMatrix
            matrix 矩阵第 column 列
    """
    m, n = matrix.shape
    return matrix[:, column - 1].reshape(m, 1).view(StpMatrix)


def stp_n(*matrix: StpMatrix) -> StpMatrix:
    """
    stp_n(matrix_1, matrix_b, matrix_c)

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
    result = matrix[0]
    for m in matrix[1:]:
        result = stp(result, m)
    return result


def array(p_object, dtype=int, *args, **kwargs) -> StpMatrix:
    """
    创建一个矩阵，使用方式和 np.array一样, 唯一的区别在于该方法返回类型为 StpMatrix
    """
    return np.array(p_object, dtype, *args, **kwargs).view(StpMatrix)


def zeros(shape, dtype=None, order='C', *args, **kwargs):
    """
    创建一个元素全为 0 的矩阵，使用方式和 np.zeros一样, 唯一的区别在于该方法返回类型为 StpMatrix
    """
    return np.zeros(shape, dtype, order, *args, **kwargs).view(StpMatrix)


def identity(n, dtype=None, *, like=None):
    """
    创建一个标准矩阵，使用方式和 np.identify一样, 唯一的区别在于该方法返回类型为 StpMatrix
    """
    return np.identity(n, dtype, like).view(StpMatrix)


class RetrieverMatrix(StpMatrix):
    """
    RetrieverMatrix(number, dimension, index)

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
    SwapMatrix(dimension_a, dimension_b)

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
                    c = i * dimension_b + j
                    r = j * dimension_a + i
                    matrix[r][c] = 1
            cls.matrix_cache[(dimension_a, dimension_b)] = matrix.view(cls)
        return cls.matrix_cache[(dimension_a, dimension_b)]


class FrontMaintainingOperator(StpMatrix):
    """
    FrontMaintainingOperator(dimension_a, dimension_b)

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
    RearMaintainingOperator(dimension_a, dimension_b)

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
    ReducingMatrix(dimension)

        降幂矩阵

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
    LogicMatrix(m, n, values)

        逻辑矩阵

        Parameters
        ----------
        m : int
            矩阵行数
        n : int
            矩阵列数
        values : list[int]
            每列对应的变量值

        Examples
        --------
        >>> LogicMatrix(2, 4, [1, 1, 2, 2])
        LogicMatirx([[1, 1, 0, 0]
                     [0, 0, 1, 1]])
    """

    def __getitem__(self, item):
        return super().__getitem__(item).view(StpMatrix)

    def __new__(cls, m, n, values: list[int]):
        matrix = zeros((m, n), dtype=int)
        matrix.fill(0)
        for i in range(n):
            matrix[values[i] - 1][i] = 1

        matrix = matrix.view(cls)
        matrix.values = values
        return matrix

    def __str__(self):
        return "LogicMatrix(" + str(self.shape[0]) + ")" + str(self.values)

    @staticmethod
    def of(matrix: StpMatrix):
        """
        LogicMatrix.logic(matrix)
            将 matrix 转换成一个逻辑矩阵
            Parameters
            ----------
            matrix : StpMatrix
        """
        m, n = matrix.shape
        values = []
        for i in range(n):
            values.append(matrix[:, i].nonzero()[0][0] + 1)
        matrix = matrix.view(LogicMatrix)
        matrix.values = values
        return matrix


class LogicValue(StpMatrix):
    """
    逻辑变量

    Parameters
    ----------
    n : int
        矩阵行数
    value : int
        该逻辑变量对应的数值
    """
    def __new__(cls, n: int, value):
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

    def __repr__(self):
        return "LogicValue[" + str(self.value) + "]"

    def __add__(self, other):
        return super().__add__(other).view(StpMatrix)
