"""
博弈相关类

作者: chang
时间: 2021/7/5
邮箱： changliangliang1996@gmail.com
"""
import itertools

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from .stp import LogicMatrix


class Strategy(object):
    """
    策略类

    Parameters
    ----------
    num : int
        策略编号，必须为正整数。

    Attributes
    ----------
    num : int
        策略编号。

    Notes
    -----
        策略编号从 1 开始。
    """

    def __init__(self, num, count):
        # TODO 策略编号需要大于0
        self.num = num
        self.count = count

    @staticmethod
    def strategy_set(n):
        """
        获得策略集合
        Parameters
        ----------
        n : 集合中的的策略个数。

        Returns
        -------
        strategies : list of strategy.
            策略集合。
        """
        strategies = []
        for x in range(n):
            strategies.append(Strategy(x + 1, n))
        return strategies

    def __str__(self):
        return "策略:" + str(self.num)


class Player(object):
    """
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
        self.strategy_set = strategy_set
        self.current_strategy = None
        self.next_strategy = None
        self.number = number
        self.current_payoff = 0
        self.next_payoff = 0

    def set_strategy(self, strategy):
        """
        设置当前策略
        Parameters
        ----------
        strategy : Strategy
            策略
        """
        self.current_strategy = strategy

    def get_strategy(self):
        """
        获得当前策略

        Returns
        -------
        strategy : Strategy
            玩家当前策略
        """
        return self.current_strategy

    def get_strategy_set(self):
        """
        玩家获得策略集

        Returns
        -------

        """

        return self.strategy_set

    def __str__(self):
        return str(self.number)


class PayOffMatrix(np.ndarray):
    """
    支付矩阵
    """

    def __new__(cls, payoff_list):
        matrix = np.array([[
            (payoff_list[0], payoff_list[1]), (payoff_list[2], payoff_list[3])],
            [(payoff_list[4], payoff_list[5]), (payoff_list[6], payoff_list[7])]], dtype=int)
        return matrix.view(cls)

    def payoff(self, player_1: Player, player_2: Player):
        """
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
        # z

    def all_payoff(self, player, graph):
        """
        玩家总收益
        Parameters
        ----------
        player : Player
            玩家
        graph : Graph
            图，保存着玩家与玩家之间的关系

        Returns
        -------
        int
            玩家总收益
        """
        # 获取玩家所有邻居
        adj_players = graph.adj_notes(player)

        # 计算总收益
        payoff = 0
        for adj_player in adj_players:
            payoff += self.payoff(player, adj_player)
        return payoff


class UpdateRule(object):
    """
    最佳响应策略
    """

    def __init__(self, graph, payoff_matrix: PayOffMatrix):
        self.graph = graph
        self.payoff_matrix = payoff_matrix

    def next_strategy(self, player):

        # 玩家当前策略
        current_strategy = player.get_strategy()

        # 玩家下一步策略
        strategy_set = player.get_strategy_set()
        next_strategy = strategy_set[0]

        player.set_strategy(next_strategy)
        temp_payoff = self.payoff_matrix.all_payoff(player, self.graph)

        for strategy in strategy_set:
            player.set_strategy(strategy)
            payoff = self.payoff_matrix.all_payoff(player, self.graph)
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

    Parameters
    ----------
    players : List[Player]
        玩家列表
    graph : Graph
        玩家关系图
    rule : UpdateRule
        玩家策略更新规则

    """

    def __init__(self, players: list[Player], graph: Graph, rule, matrix: PayOffMatrix):
        self.rule = rule
        self.payoff_matrix = matrix
        self.graph = graph
        self.players: list[Player] = players

    def payoff(self, player):
        """
        获得指定玩家总收益
        :param player: 指定玩家
        :return: 总收益
        """

        # 获取玩家所有邻居
        adj_players = self.graph.adj_notes(player)

        # 计算总收益
        payoff = 0
        for adj_player in adj_players:
            payoff += self.payoff_matrix.payoff(player, adj_player)
        return payoff

    def update_payoff(self):
        """
        更新玩家当前收益
        Returns
        -------
        """
        # 更新玩家收益
        for player in self.players:
            payoff = self.payoff(player)
            player.payoff = payoff

    def set_strategies(self, strategies: list[Strategy]):
        """
        设置博弈中所有玩家策略

        Parameters
        ----------
        strategies

        Returns
        -------

        """

        players = self.players
        for x in range(len(players)):
            players[x].set_strategy(strategies[x])

    def strategy(self, player):
        """
        获得玩家下一步的策略
        Parameters
        ----------
        player

        Returns
        -------

        """
        return self.rule.next_strategy(player)

    def update_strategies(self):
        r = []
        for player in self.players:
            self.strategy(player)
            r.append(player.next_strategy.num + 1)
        return r

    def set_profile(self, profile):
        """
        设置当前局势
        Parameters
        ----------
        profile : tuple[Strategy]
            局势
        """
        for i in range(len(profile)):
            player = self.players[i]
            player.set_strategy(profile[i])

    def struct_matrix_player(self, player: Player) -> LogicMatrix:

        # 获得所有可能的局势
        strategy_set_list = []
        for player_ in self.players:
            strategy_set_list.append(player_.get_strategy_set())
        profiles = itertools.product(*strategy_set_list)

        next_strategies = []
        for profile in profiles:
            self.set_profile(profile)

            next_strategies.append(self.strategy(player))
        values = []
        m = player.current_strategy.count
        for strategy in next_strategies:
            values.append(strategy.num)
        return LogicMatrix(m, len(values), values)

    def struct_matrix(self):
        """
        获得网络演化博弈的结构矩阵

        Returns
        -------
        core.stp.LogicMatrix
            网络演化博弈的结构矩阵
        """

        # 获得所有可能的局势
        strategy_set_list = []
        for player in self.players:
            strategy_set_list.append(player.get_strategy_set())
        profiles = itertools.product(*strategy_set_list)

        # 下一步局势集合
        next_profiles = []
        for profile in profiles:
            self.set_profile(profile)

            next_profile = []
            for player in self.players:
                next_profile.append(self.strategy(player))
            next_profiles.append(next_profile)

        values = []
        m = 1
        for profile in next_profiles:
            m = 1
            value = 0
            for strategy in profile:
                m = m * strategy.count
                value = value * strategy.count + strategy.num - 1
            value = value + 1
            values.append(value)
        return LogicMatrix(m, len(values), values=values)
