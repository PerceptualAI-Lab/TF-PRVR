import numpy as np

from features import normalize, text_features, correct_queries
from segmentation import temporal_signal, frequency_boundaries, make_segments
from graph import build_graph, propagate


def moment_features(nodes, spans, offsets, duration_power, norm_power):
    edges = np.unique(spans)
    midpoints = (edges[:-1] + edges[1:]) / 2
    covering = []

    for start, end in zip(offsets[:-1], offsets[1:]):
        current = spans[start:end]
        contains = (
            (current[:, 0, None] <= midpoints[None, :])
            & (midpoints[None, :] < current[:, 1, None])
        )

        if not np.all(contains.sum(axis=0) == 1):
            raise ValueError("Each scale must partition the full video.")

        covering.append(start + contains.argmax(axis=0))

    covering = np.column_stack(covering)
    lengths = spans[covering, 1] - spans[covering, 0]
    weights = lengths.astype(float) ** (-duration_power)
    weights = weights / weights.sum(axis=1, keepdims=True)

    moments = np.sum(weights[:, :, None] * nodes[covering], axis=1)

    return normalize(moments, power=norm_power)


def video_features(frames, segmentation, graph, mas, beta):
    nodes, spans, offsets = make_segments(frames, **segmentation)
    adjacency = build_graph(nodes, spans, offsets, **graph)

    nodes = propagate(nodes, adjacency, beta, mas.get("node_norm_power", 1))
    moments = moment_features(
        nodes, spans, offsets, mas["duration_power"], mas.get("moment_norm_power", 1),
    )

    return moments


def score_video(queries, moments, top_k):
    if top_k < 1 or not isinstance(top_k, (int, np.integer)):
        raise ValueError("top_k must be a positive integer.")

    queries = np.asarray(queries, dtype=np.float64)
    moments = np.asarray(moments, dtype=np.float64)
    scores = queries @ moments.T
    count = min(top_k, len(moments))

    if count == 1:
        return scores.max(axis=1)

    split = scores.shape[1] - count
    best = np.partition(scores, split, axis=1)[:, split:]

    return best.mean(axis=1)
