

from stp.game import *
from stp import *

# 策略集
strategy_set = Strategy.strategy_set(2)

# 玩家
player_1 = Player(1, strategy_set)
player_2 = Player(2, strategy_set)
player_3 = Player(3, strategy_set)
player_4 = Player(4, strategy_set)

# 网络图
graph = Graph()
graph.add_node(player_1).add_node(player_2).add_node(player_3).add_node(player_4)
graph.add_edge(player_1, player_2).add_edge(player_1, player_4).add_edge(player_2, player_3).add_edge(player_3, player_4)
# graph.draw(labels={player_1: 1, player_2: 2, player_3: 3})

# 囚徒博弈支付矩阵
payoff_matrix = PayOffMatrix([1, 2, 0, 0, 0, 0, 2, 1])

# 网络演化博弈
game = Game([player_1, player_2, player_3, player_4], graph, UpdateRule(graph, payoff_matrix), payoff_matrix)

game.set_strategies([strategy_set[0], strategy_set[0], strategy_set[0], strategy_set[1]])

a = khatri_rao(game.struct_matrix_player(player_2), game.struct_matrix_player(player_3))
a = khatri_rao(a, game.struct_matrix_player(player_4))
# print(LogicMatrix.as_logic_matrix(a))
print(game.struct_matrix())

L2 = game.struct_matrix_player(player_2)
L3 = game.struct_matrix_player(player_3)
L4 = game.struct_matrix_player(player_4)

L = L2.khatri_rao(L3).khatri_rao(L4)
print(LogicMatrix.as_logic_matrix(L))

for i, value in enumerate(L.values):
    print(i + 1, "====", value)
