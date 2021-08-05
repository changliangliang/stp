from stp.game import *
from stp import *


def test_game():

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
    graph.add_edge(player_1, player_2).add_edge(player_1, player_4).add_edge(player_2, player_3).add_edge(player_3,
                                                                                                          player_4)

    # 支付矩阵
    payoff_matrix = PayOffMatrix([1, 2, 0, 0, 0, 0, 2, 1])

    # 网络演化博弈
    game = Game([player_1, player_2, player_3, player_4], graph, UpdateRule(graph, payoff_matrix), payoff_matrix)
    assert (game.struct_matrix() == LogicMatrix(16, 16, [1, 11, 6, 16, 11, 11, 16, 16, 6, 16, 6, 16, 16, 16, 16, 16])).all()
