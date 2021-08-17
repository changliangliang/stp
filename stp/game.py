"""
与网络演化博弈相关类


作者: chang
时间: 2021/7/5
邮箱： changliangliang1996@gmail.com
"""
import itertools

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from .stp import LogicMatrix, array, SwapMatrix


class Strategy(object):
    """
    Strategy(num, count)

        策略类，代表每个玩家选择的策略

        Parameters
        ----------
        num : int
            策略编号，必须为正整数。
        count : int
            策略总数
    """

    def __init__(self, num, count):
        # TODO 策略编号需要大于0
        self.num = num
        self.count = count

    @staticmethod
    def set(n):
        """
        Strategy.set(n)

            获得一个由 n 个策略组成到的策略集

            Parameters
            ----------
            n : int
                策略集中策略个数

            Returns
            -------
            List[Strategy]
                策略列表
        """
        strategies = []
        for x in range(n):
            strategies.append(Strategy(x + 1, n))
        return strategies

    def __str__(self):
        return "Strategy[" + str(self.num) + "]"


class Player(object):
    """
    Player(number, strategy_set)

        玩家类

        Parameters
        ----------
        number : int
            玩家编号。
        strategy_set : list of strategy
            玩家策略集。

        Attributes
        ----------
        number : int
            玩家编号
        strategy_set : list of strategy
            玩家策略集
        current_strategy : strategy
            玩家当前策略.
        next_ strategy : strategy
            玩家下一步策略.
        current_payoff : int
            玩家当前收益.
        next_payoff : int
            玩家下一步收益
    """

    def __init__(self, number, strategy_set: list):

        # 玩家编号和玩家策略集
        self.number = number
        self.strategy_set = strategy_set

        # 记录玩家状态
        self.current_strategy = None
        self.next_strategy = None
        self.current_payoff = 0
        self.next_payoff = 0

    def set_strategy(self, strategy):
        """
        设置当前策略
        """
        self.current_strategy = strategy

    def get_strategy(self):
        """
        获得当前策略
        """
        return self.current_strategy

    def get_strategy_set(self):
        """
        玩家获得策略集
        """
        return self.strategy_set

    def __str__(self):
        return str(self.number)


class PayOffMatrix(np.ndarray):
    """
    PayOffMatrix(payoff_list)

        支付矩阵

    Examples
    --------
    >>> PayOffMatrix([1, 2, 3, 4, 5, 6, 7, 8])
    [[(1, 2), (3, 4)]
     [(5, 6), (7, 8)]]
    """

    def __new__(cls, payoff_list):
        matrix = np.array([[
            (payoff_list[0], payoff_list[1]), (payoff_list[2], payoff_list[3])],
            [(payoff_list[4], payoff_list[5]), (payoff_list[6], payoff_list[7])]], dtype=int)
        return matrix.view(cls)

    def payoff(self, player_1: Player, player_2: Player):
        """
        p.payoff(player_1, player_2)

            基础网路博弈收益

            Parameters
            ----------
            player_1 : Player
                玩家1
            player_2 : Player
                玩家2

            Returns
            -------
            int
                玩家1和玩家2之间博弈后玩家1的收益
        """
        return self[player_1.current_strategy.num - 1][player_2.current_strategy.num - 1][0]
        # TODO 策略编号从 1 开始，后续应该修改


class UpdateRule(object):
    """
    最佳响应策略
    """

    def __init__(self):
        self.game = None

    def next_strategy(self, player):

        # 玩家当前策略
        current_strategy = player.get_strategy()

        # 玩家下一步策略
        strategy_set = player.get_strategy_set()
        next_strategy = strategy_set[0]

        player.set_strategy(next_strategy)
        temp_payoff = self.game.payoff(player)

        for strategy in strategy_set:
            player.set_strategy(strategy)
            payoff = self.game.payoff(player)
            if payoff > temp_payoff:
                temp_payoff = payoff
                next_strategy = strategy
        player.next_strategy = next_strategy
        player.current_strategy = current_strategy
        return next_strategy


class Graph(object):

    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, node):
        self.graph.add_node(node)
        return self

    def add_edge(self, node_1, node_2):
        self.graph.add_edge(node_1, node_2)
        return self

    def adj_notes(self, node):
        nodes = list(self.graph.adj[node].keys())
        return nodes

    def all_nodes(self):
        return list(self.graph.nodes)

    def draw(self, labels=None):
        nx.draw(self.graph, with_labels=True, labels=labels)
        plt.show()


class Game(object):
    """
    网路演化博弈
    """

    def __init__(self):
        self.rule = None
        self.matrix = None
        self.graph = None

    def set_graph(self, graph):
        self.graph = graph
        graph.game = self

    def set_matrix(self, matrix):
        self.matrix = matrix
        matrix.game = self

    def set_rule(self, rule: UpdateRule):
        self.rule = rule
        rule.game = self

    def players(self):
        """
        获得所有玩家
        """
        return self.graph.all_nodes()

    def payoff(self, player):
        """
        获得某个玩家总收益
        """

        # 获取玩家所有邻居
        adj_players = self.graph.adj_notes(player)

        # 计算总收益
        payoff = 0
        for adj_player in adj_players:
            payoff += self.matrix.payoff(player, adj_player)
        return payoff

    def set_strategies(self, strategies: list[Strategy]):
        """
        设置博弈中所有玩家策略
        """
        players = self.players()
        for x in range(len(players)):
            players[x].set_strategy(strategies[x])

    def next_strategy(self, player):
        """
        获得玩家下一步的策略
        """
        return self.rule.next_strategy(player)

    def set_profile(self, profile):
        """
        设置当前局势
        """
        for i in range(len(profile)):
            player = self.players()[i]
            player.set_strategy(profile[i])

    def struct_matrix_player(self, player: Player) -> LogicMatrix:

        # 获得所有可能的局势
        strategy_set_list = []
        for player_ in self.players():
            strategy_set_list.append(player_.get_strategy_set())
        profiles = itertools.product(*strategy_set_list)

        next_strategies = []
        for profile in profiles:
            self.set_profile(profile)

            next_strategies.append(self.next_strategy(player))
        values = []
        m = player.current_strategy.count
        for strategy in next_strategies:
            values.append(strategy.num)
        return LogicMatrix(m, len(values), values)

    def struct_matrix(self):
        matrix = array([[1]])
        for player in self.players():
            matrix = matrix.stp(self.struct_matrix_player(player))
        return GameStructMatrix.of(matrix)


class GameStructMatrix(LogicMatrix):
    """
    结构矩阵
    """

    def __new__(cls, *args, **kwargs):

        obj = super().__new__(LogicMatrix, *args, **kwargs)
        return obj.view(GameStructMatrix)

    @staticmethod
    def of(matrix):
        m, n = matrix.shape
        values = []
        for i in range(n):
            values.append(matrix[:, i].nonzero()[0][0] + 1)
        matrix = matrix.view(GameStructMatrix)
        matrix.values = values
        return matrix

    def set_ctrl_stat(self, w, z):
        """
        设置控制和状态的维数
        """
        matrix = self.stp(SwapMatrix(w, z))
        setattr(self, "matrix", matrix)

    def is_reachable(self, from_stat, to_stat):
        """
        判断是否能从 from_stat 到达 to_stat
        """
        matrix = getattr(self, "matrix", self)
        matrix = matrix.stp(from_stat)
        matrix = np.sum(matrix, axis=1).reshape((matrix.shape[0], 1))
        matrix = np.logical_and(matrix, to_stat)
        return matrix.any()

    def get_reachable_set(self, stat, stats: list):
        """
        获得可达集
        """
        # 可达集
        reachable_sets = [[stat]]
        stats.remove(stat)

        # 全局可达标志
        global_reachable = True

        # 还有状态
        while len(stats) > 0:

            if not global_reachable:
                return None
            global_reachable = False

            to_stat = reachable_sets[-1][0]
            for stat in reachable_sets[-1][1:]:
                to_stat = to_stat + stat
            reachable_set = []
            for stat in stats[:]:
                # 如果可达
                if self.is_reachable(stat, to_stat):
                    reachable_set.append(stat)
                    stats.remove(stat)

                    # 目前还是全局可达
                    global_reachable = True

            reachable_sets.append(reachable_set)
        return reachable_sets
