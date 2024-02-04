import os

from logging import getLogger
from uuid import uuid4

import torch

from matplotlib import pyplot as plt

from solvers.concorde_solver import concorde_solve

logger = getLogger(__name__)


def generate_random_graph(
    num_nodes: int = 10,
    render: bool = False,
) -> torch.Tensor:
    """
    Generate a random set of points of a given size.
    """
    points: torch.Tensor = (torch.rand((2, num_nodes)) - 0.5) * 2.0

    if render:
        plt.scatter(*points.numpy())
        plt.show()

    return points


def generate_dataset(
    num_nodes: int,
    num_graphs: int,
    directory: str = "data",
) -> None:
    """
    Try to generate a dataset with a given number of graphs of a given size.
    """
    try:
        import concorde
    except ImportError:
        print("Can't generate datasets on Windows. Use WSL.")
        return

    try:
        os.mkdir(f"data/{directory}_{num_nodes}_{num_graphs}")
        logger.info(f"Created directory data/{directory}_{num_nodes}_{num_graphs}. Populating.")
    except FileExistsError:
        logger.info(
            f"Don't need to create directory data/{directory}_{num_nodes}_{num_graphs} as it exists. Returning."
        )
        return

    for _ in range(num_graphs):
        with open(f"data/{directory}_{num_nodes}_{num_graphs}/{uuid4()}.tsp", "w+") as file:
            points = generate_random_graph(num_nodes, render=False)
            solution = concorde_solve(points)
            file.write(",".join([str(int(s)) for s in solution]))
            file.write("\n")

            for x, y in points.t():
                file.write(f"{x} {y}\n")

    logger.info("Done creating dataset.")


def import_dataset(
    num_nodes: int,
    num_graphs: int,
    directory: str = "data",
) -> tuple[list[torch.Tensor], list[list[int]]]:
    """
    Try to import a previously created dataset.
    """
    dir_name: str = f"data/{directory}_{num_nodes}_{num_graphs}"
    contents = os.listdir(dir_name)

    problems: list[list[tuple[float, float]]] = []
    solutions: list[list[int]] = []
    for filename in contents:
        with open(f"{dir_name}/{filename}", "r") as file:
            solution_string: str = file.readline().strip()
            points_strings: list[str] = [line.strip() for line in file.readlines()]

        solution: list[int] = [int(x) for x in solution_string.split(",")]
        points: list[tuple[float, float]] = [(float(x), float(y)) for x, y in [line.split() for line in points_strings]]

        problems.append(points)
        solutions.append(solution)

    return [torch.tensor(problem).t() for problem in problems], solutions
