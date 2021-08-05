"""
各种矩阵

作者: chang
时间: 2021/1/26
邮箱： changliangliang1996@gmail.com
"""

# class DescendingPowerMatrix(np.ndarray):
#     """
#     降幂矩阵
#     """
#
#     def __new__(cls, l, k):
#         base_reducing_matrix = BaseReducingMatrix(k)
#         identity_k = np.identity(k, dtype=int)
#         result_matrix = np.array([[1]], dtype=int)
#
#         for i in range(1, l + 1):
#             result_matrix = stp(result_matrix,
#                                 np.kron(np.identity(np.power(k, i - 1), dtype=int),
#                                         stp(np.kron(identity_k, SwapMatrix(k, np.power(k, l - i))),
#                                             base_reducing_matrix)))
#         return result_matrix.view(cls)



