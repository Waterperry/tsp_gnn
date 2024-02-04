from typing import Type

import torch

from matplotlib import pyplot as plt


Route: Type = torch.Tensor


def compute_edge_length(points: torch.Tensor, u: int, v: int) -> torch.Tensor:
    """
    Utility function to calculate the distance between a node index u
        and another node index v given a set of points.
    """
    return torch.dist(points[0:2, u], points[0:2, v])


def compute_route_length(points: torch.Tensor, route: Route) -> torch.Tensor:
    """
    Calculate the total tour length of a given route on some points.
    """

    # append the first node of the route onto the end to calculate the full trip's length.
    route = torch.concat((route, route[0][None]))

    x, y = points[0], points[1]
    dx = x[route[1:]] - x[route[:-1]]
    dy = y[route[1:]] - y[route[:-1]]
    distances = torch.sqrt(dx**2 + dy**2)

    tour_length = torch.sum(distances)

    return tour_length


def plot_route(
    points: torch.Tensor,
    route: list[int] | torch.Tensor,
    show: bool = True,
    plot_return_line: bool = True,
) -> None:
    """
    Plot a set of points and a given route.
    """
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.scatter(*points, s=29)

    for idx, (u, v) in enumerate(zip(route[:-1], route[1:])):
        _xs = float(points[0, u]), float(points[0, v])
        _ys = float(points[1, u]), float(points[1, v])
        plt.plot(_xs, _ys, color="green")

    if plot_return_line:
        # plot final return line
        src_x, src_y = points[:, route[-1]]
        dst_x, dst_y = points[:, route[0]]
        plt.plot(
            [float(src_x), float(dst_x)],
            [float(src_y), float(dst_y)],
            color="green",
        )

    if show:
        plt.show()


def plot_loss(
    training_losses: list[torch.Tensor],
    validation_losses: list[torch.Tensor] | None = None,
    min_validation_losses: list[torch.Tensor] | None = None,
) -> None:
    """
    Plot the training, validation, and minimum observed validation losses for each epoch during training.
    """
    plt.plot(
        [training_loss.detach().numpy() for training_loss in training_losses],
        label="Training Loss",
        color="blue",
    )
    plt.plot(
        [validation_loss.detach().numpy() for validation_loss in validation_losses],
        label="Val. Loss",
        color="orange",
    )
    plt.plot(
        min_validation_losses,
        label="Min. Val. Loss",
        linestyle="dashed",
        color="orange",
    )
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.show()
