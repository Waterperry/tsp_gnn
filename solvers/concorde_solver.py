import torch

from utils import Route, compute_route_length


def concorde_solve(points: torch.Tensor) -> Route | None:
    """
    Solve a TSP using Concorde (requires Linux/WSL and pyconcorde!)
    """
    from concorde.tsp import TSPSolver

    scaler: float = 1e6

    xs, ys = points[0], points[1]

    solver = TSPSolver.from_data(xs * scaler, ys * scaler, norm="EUC_2D")
    solution = solver.solve(verbose=False)
    route_list: list[int] = solution.tour

    route: torch.Tensor = torch.tensor(route_list)

    return route
