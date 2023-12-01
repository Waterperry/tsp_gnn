import random

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np


class TravellingSalesmanProblem:
    def __init__(self, random: bool = False) -> None:
        if random:
            self.from_random()
            return

        self.num_states = 5
        self.density = -1
        self._edges = np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [1, 0, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
            ],
        )

        self._weights = np.array(
            [
                [0, 5, 2, 0, 0],
                [5, 0, 0, 0, 3],
                [2, 0, 0, 0, 0],
                [0, 0, 0, 0, 6],
                [0, 3, 0, 6, 0],
            ],
        )

    def from_random(self, num_states: int = 52, density: float = 0.1) -> None:
        self.num_states = num_states
        self.density = density
        self._edges = np.zeros((num_states, num_states), dtype=np.int8)
        self._weights = np.zeros((num_states, num_states), dtype=np.int8)

        for i in range(num_states):
            for j in range(num_states):
                if i == j:
                    self._edges[i][j] = 0
                    self._weights[i][j] = -1
                    continue

                is_edge: int = int(random.random() < density)
                self._edges[i][j] = is_edge
                self._edges[j][i] = is_edge

                weight: int = random.randint(1, 5) if is_edge else -1
                self._weights[i][j] = weight
                self._weights[j][i] = weight

    def show_graph(self) -> None:
        graph: nx.DiGraph = nx.DiGraph()

        node_labels: dict[int, str] = {}
        for i in range(self.num_states):
            node_labels[i] = f"{i}"
            graph.add_node(i)

        edge_labels: dict[tuple[int, int], str] = {}
        for i in range(self.num_states):
            for j in range(self.num_states):
                weight = self._weights[i][j]
                edge_labels[(i, j)] = f"{weight}" if weight else ""
                if self._edges[i][j]:
                    graph.add_edge(i, j)

        pos = nx.circular_layout(graph)
        nx.draw_networkx_edge_labels(
            graph,
            pos=pos,
            edge_labels=edge_labels,
        )
        nx.draw_networkx_labels(graph, pos, node_labels)
        nx.draw(graph, pos=pos)
        plt.show()
