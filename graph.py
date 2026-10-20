import numpy as np
from scipy.linalg import solve

from features import normalize


def build_graph(nodes, spans, offsets, knn, segment_knn, similarity_power,
                temporal_weight, tau_scale, inter_weight):
    if tau_scale <= 0:
        raise ValueError("tau_scale must be positive.")

    visual = np.maximum(nodes @ nodes.T, 0) ** similarity_power
    weights = np.zeros_like(visual)

    for level in range(len(offsets) - 1):
        start, end = offsets[level:level + 2]
        current = spans[start:end]

        centers = current.mean(axis=1)
        duration = (current[:, 1] - current[:, 0]).mean()
        separation = np.abs(centers[:, None] - centers[None, :])
        temporal = np.exp(-separation / (duration * tau_scale))

        block = visual[start:end, start:end] + temporal_weight * temporal
        np.fill_diagonal(block, 0)

        neighbors = knn if level == 0 else segment_knn

        if neighbors > 0 and len(current) > neighbors + 1:
            dropped = np.argsort(-block, axis=1)[:, neighbors:]
            np.put_along_axis(block, dropped, 0, axis=1)
            block = (block + block.T) / 2

        weights[start:end, start:end] = block

        if level == len(offsets) - 2:
            continue

        next_end = offsets[level + 2]
        adjacent = spans[end:next_end]

        left = np.maximum(current[:, None, 0], adjacent[None, :, 0])
        right = np.minimum(current[:, None, 1], adjacent[None, :, 1])
        overlap = np.maximum(right - left, 0)

        current_length = (current[:, 1] - current[:, 0])[:, None]
        adjacent_length = (adjacent[:, 1] - adjacent[:, 0])[None, :]
        overlap = overlap / np.sqrt(current_length * adjacent_length)

        cross = overlap * visual[start:end, end:next_end] * inter_weight
        weights[start:end, end:next_end] = cross
        weights[end:next_end, start:end] = cross.T

    np.fill_diagonal(weights, 1)
    degree = weights.sum(axis=1)

    return weights / degree[:, None]


def propagate(nodes, adjacency, beta, norm_power):
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1).")

    system = np.eye(len(nodes)) - beta * adjacency
    propagated = (1 - beta) * solve(system, nodes, assume_a="gen")

    return normalize(propagated, power=norm_power)
