import torch

from utils import Route, compute_edge_length


def closest_node_solve(points: torch.Tensor) -> Route:
    """
    Solve a TSP by choosing to visit the nearest point we have not yet visited.
    """
    num_nodes: int = points.shape[1]

    route: list[int] = []
    curr: int = 0
    while len(route) != num_nodes:
        route.append(curr)

        min_dist: torch.Tensor = torch.tensor(9e9)
        min_i: int = -1
        for i in range(num_nodes):
            if i == curr:
                continue
            if i in route:
                continue

            dist: torch.Tensor = compute_edge_length(points, i, curr)
            if min_dist > dist:
                min_dist = dist
                min_i = i

        curr = min_i

    route_tensor: torch.Tensor = torch.tensor(route)
    return route_tensor
