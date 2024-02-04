from collections import deque
from collections.abc import Callable

import torch
from matplotlib import pyplot as plt
from numpy import sqrt

from utils import compute_route_length, plot_route


def beam_search_decode(
    points: torch.Tensor,
    output: torch.Tensor,
    beam_width_mapping: Callable[[int], [int]] = lambda r_len: 3 if r_len > 7 else 2,
    render: bool = False,
    render_prefix: str = "",
) -> list[int]:
    """
    Use beam search decoding to decode the neural network's output.

    :param points: problem specification ([x,y] coords for N nodes)
    :param output: (N^2, 1) probability matrix from neural network (or other source)
    :param beam_width_mapping: function mapping a route length onto a beam width
    :param render: Whether to render the output to the render/beam-solve directory for animation
    :param render_prefix: Prefix to save rendered .png files with
    :return: the beam-search decoded optimal route according to the given probability matrix
    """
    num_nodes = int(sqrt(output.shape[0]))
    reshaped_output = output.reshape((num_nodes, num_nodes))

    routes: deque[list[int]] = deque()
    final_routes: list[torch.Tensor] = []
    routes.append([0])

    cached_argsort: list[list[int]] = [
        torch.argsort(reshaped_output[i], descending=True).tolist() for i in range(num_nodes)
    ]

    idx: int = 0
    while True:
        if not routes:
            break

        route: list[int] = routes.popleft()
        route_length: int = len(route)

        if route_length == num_nodes:
            if render:
                plot_route(points, route, show=False, plot_return_line=True)
                plt.savefig(f"render/beam-solve/{render_prefix}partial_{idx:0>5}.png")
                plt.clf()
                idx += 1

            final_routes.append(torch.tensor(route))
            continue

        # our current node is the last node on the route so far
        u: int = route[-1]
        vs: list[int] = cached_argsort[u]
        beams: set[int] = set()

        # for each potential destination, if we haven't already been there, add a beam going to it
        for potential_v in vs:
            if potential_v not in route:
                beams.add(potential_v)

            if len(beams) >= beam_width_mapping(route_length + 1):
                break

        # add the route + beam to our current working routes
        for beam in beams:
            new_route = route + [beam]
            if render:
                plot_route(points, new_route, show=False, plot_return_line=False)
                plt.savefig(f"render/beam-solve/{render_prefix}partial_{idx:0>5}.png")
                plt.clf()
                idx += 1

            routes.append(new_route)

    # return the shortest route
    shortest_route = min(final_routes, key=lambda rt: compute_route_length(points, rt)).tolist()

    if render:
        plot_route(points, shortest_route, show=False, plot_return_line=True)
        # render 30 frames of the final route
        for d_idx in range(1, 31):
            plt.savefig(f"render/beam-solve/{render_prefix}partial_{idx+d_idx:0>5}.png")
        plt.clf()

    return shortest_route
