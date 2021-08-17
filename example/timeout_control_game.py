import numpy as np

from stp.game import *
from stp import *

# 策略集
strategy_set = Strategy.set(2)

# 玩家
player_1 = Player(1, strategy_set)
player_2 = Player(2, strategy_set)
player_3 = Player(3, strategy_set)
player_4 = Player(4, strategy_set)

# 网络图
graph = Graph()
graph.add_node(player_1).add_node(player_2).add_node(player_3).add_node(player_4)
graph.add_edge(player_1, player_2).add_edge(player_1, player_4).add_edge(player_2, player_3).add_edge(player_3, player_4)


# 支付矩阵
payoff_matrix = PayOffMatrix([2, 2, 1, 0, 0, 1, 3, 3])

rule = UpdateRule()

# 网络演化博弈
game = Game()
game.set_rule(rule)
game.set_graph(graph)
game.set_matrix(payoff_matrix)

# 获得玩家2、3、4更新规则结构矩阵
L2 = game.struct_matrix_player(player_2)
L3 = game.struct_matrix_player(player_3)
L4 = game.struct_matrix_player(player_4)

print("==", L2)
print("==", L3)
print("==", L4)

# 获得总体结构矩阵
L = L2.khatri_rao(L3).khatri_rao(L4)
print(LogicMatrix.of(L))

L = GameStructMatrix.of(L)
L.set_ctrl_stat(2, 8)


# 计算可达集
b = LogicMatrix.of(L.stp(SwapMatrix(2, 8)))
to_stat = LogicValue(8, 1)
stats = [to_stat, LogicValue(8, 2), LogicValue(8, 3), LogicValue(8, 4), LogicValue(8, 5), LogicValue(8, 6), LogicValue(8, 7), LogicValue(8, 8)]
reachable_sets = L.get_reachable_set(to_stat, stats)
print("可达集==>", reachable_sets)
print("L==>", L)

# 以2步延时
M = L.stp(
    np.kron(
        np.identity(2, dtype=int),
        L
    ).view(StpMatrix)
)
print("M==>")
print(M)


def set_add(set_):
    result = set_[0]
    for s in set_[1:]:
        result = result + s
    return result


def is_belong(stat, r_set):
    print(r_set)
    return np.logical_and(stat, np.sum(r_set)).all()


H = zeros((2, 32), dtype=int)

for i in range(32):
    stat = M[:, i].reshape(8, 1)
    if stat in reachable_sets[0]:
        H[0][i] = 1
        continue

    if stat in reachable_sets[1]:
        if L.stp(LogicValue(2, 2)).stp(stat) in reachable_sets[0]:
            H[1][i] = 1
        else:
            H[0][i] = 1
        continue
    if stat in reachable_sets[2]:
        if L.stp(LogicValue(2, 2)).stp(stat) in reachable_sets[1]:
            H[1][i] = 1
        else:
            H[0][i] = 1

print(LogicMatrix.of(H))
fai_a = L.stp(
    np.kron(
        np.identity(2, dtype=int),
        L
    ).view(StpMatrix)
)


t = L.stp(H).stp(
    np.kron(
        np.identity(32),
        fai_a,
    ).view(StpMatrix)
)

t = t.stp(ReducingMatrix(32))


fai_b = H.stp(
    np.kron(
        np.identity(64, dtype=int),
        L
    ).view(StpMatrix)
).stp(ReducingMatrix(32))

print("b=====")
print(fai_b.shape)
print(LogicMatrix.of(fai_b))
c = t.stp(np.linalg.matrix_power(fai_b, 10)).astype(int)
print(LogicMatrix.of(c))
