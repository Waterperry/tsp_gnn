import torch

from numpy import sqrt


def naive_decode(output: torch.Tensor) -> list[int]:
    """
    Decode an output probability tensor "naively", by starting at node 0 and picking the
    highest-ranked node repeatedly until a full route is produced.
    """
    num_nodes: int = int(sqrt(output.shape[0]))
    reshaped_output = output.reshape((num_nodes, num_nodes))

    route = []
    u = 0
    for _ in range(num_nodes):
        route.append(u)
        vs: list[int] = torch.argsort(reshaped_output[u], descending=True).tolist()
        v = -1
        for potential_v in vs:
            if potential_v not in route:
                v = potential_v
                break
        u = v

    return route
