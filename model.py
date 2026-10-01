from dataclasses import dataclass

import numpy as np
import pywt
from scipy.linalg import solve
from scipy.signal import find_peaks


@dataclass
class Hierarchy:
    features: np.ndarray
    spans: np.ndarray
    offsets: np.ndarray


def normalize(features):
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return np.divide(features, norms, out=np.zeros_like(features), where=norms > 1e-12)


def validate_features(features, ndim, name):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != ndim or any(size == 0 for size in features.shape):
        raise ValueError(f"{name}: expected a nonempty {ndim}D feature array")
    if not np.isfinite(features).all():
        raise ValueError(f"{name}: features contain NaN or infinity")
    if np.any(np.linalg.norm(features, axis=-1) < 1e-12):
        raise ValueError(f"{name}: zero feature vectors are not valid embeddings")
    return features


def segment_video(frames, wavelet="db4"):
    frames = normalize(validate_features(frames, 2, "video"))
    basis = pywt.Wavelet(wavelet)
    length = len(frames)
    cuts_by_level = []
    if length >= 3:
        velocity = np.diff(frames, axis=0)
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        velocity /= speed + 1e-8
        signal = 1.0 - np.clip(np.sum(velocity[:-1] * velocity[1:], axis=1), -1, 1)
        stationary = speed[:, 0] <= 1e-8
        signal[stationary[:-1] & stationary[1:]] = 0.0
        depth = pywt.dwt_max_level(len(signal), basis.dec_len)
        if depth:
            coefficients = pywt.wavedec(signal, basis, mode="symmetric", level=depth)
            tolerance = 64 * np.finfo(np.float64).eps * max(1.0, np.max(np.abs(signal)))
            for level in range(1, depth + 1):
                isolated = [np.zeros_like(coefficient) for coefficient in coefficients]
                isolated[-level] = coefficients[-level]
                response = np.abs(pywt.waverec(isolated, basis, mode="symmetric")[:len(signal)])
                peaks = find_peaks(response)[0]
                threshold = response.mean() + response.std()
                peaks = peaks[response[peaks] > max(threshold, tolerance)]
                cuts_by_level.append(peaks + 2)
    if not cuts_by_level:
        cuts_by_level = [np.empty(0, dtype=np.int64)]
    features, spans, offsets = [], [], [0]
    cumulative = np.vstack([np.zeros((1, frames.shape[1])), np.cumsum(frames, axis=0)])
    for cuts in cuts_by_level:
        edges = np.unique(np.r_[0, cuts, length]).astype(np.int64)
        starts, ends = edges[:-1], edges[1:]
        pooled = (cumulative[ends] - cumulative[starts]) / (ends - starts)[:, None]
        features.append(normalize(pooled))
        spans.append(np.column_stack([starts, ends]))
        offsets.append(offsets[-1] + len(starts))
    return Hierarchy(np.concatenate(features), np.concatenate(spans), np.asarray(offsets))


def build_graph(hierarchy):
    features, spans, offsets = hierarchy.features, hierarchy.spans, hierarchy.offsets
    weights = np.zeros((len(features), len(features)), dtype=np.float64)
    for level, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        local = features[start:end]
        centers = spans[start:end].mean(axis=1)
        duration = np.diff(spans[start:end], axis=1).mean()
        temporal = np.exp(-np.abs(centers[:, None] - centers[None, :]) / duration)
        weights[start:end, start:end] = np.maximum(local @ local.T, 0) + temporal
        if level + 2 < len(offsets):
            next_end = offsets[level + 2]
            current_spans = spans[start:end]
            next_spans = spans[end:next_end]
            intersection = np.maximum(
                np.minimum(current_spans[:, None, 1], next_spans[None, :, 1])
                - np.maximum(current_spans[:, None, 0], next_spans[None, :, 0]), 0,
            )
            union = (
                np.diff(current_spans, axis=1)
                + np.diff(next_spans, axis=1).T - intersection
            )
            cross = intersection / union * np.maximum(local @ features[end:next_end].T, 0)
            weights[start:end, end:next_end] = cross
            weights[end:next_end, start:end] = cross.T
    np.fill_diagonal(weights, 1.0)
    inv_sqrt_degree = 1.0 / np.sqrt(weights.sum(axis=1))
    return inv_sqrt_degree[:, None] * weights * inv_sqrt_degree[None, :]


def propagation_kernel(graph, beta=0.7):
    if not np.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("beta must satisfy 0 <= beta < 1")
    identity = np.eye(len(graph))
    return (1 - beta) * solve(identity - beta * graph, identity, assume_a="pos")


def compile_video(frames, wavelet="db4", beta=0.7):
    hierarchy = segment_video(frames, wavelet)
    kernel = propagation_kernel(build_graph(hierarchy), beta)
    propagated = kernel @ hierarchy.features
    edges = np.unique(hierarchy.spans)
    midpoints = (edges[:-1] + edges[1:]) / 2
    moments = np.zeros((len(midpoints), propagated.shape[1]))
    for start, end in zip(hierarchy.offsets[:-1], hierarchy.offsets[1:]):
        nodes = start + np.searchsorted(hierarchy.spans[start:end, 1], midpoints, side="right")
        moments += propagated[nodes]
    moments /= len(hierarchy.offsets) - 1
    return moments, np.column_stack([edges[:-1], edges[1:]]), hierarchy


def score_video(queries, moments, block_size=4096):
    if block_size < 1:
        raise ValueError("block_size must be positive")
    best = np.full(len(queries), -np.inf)
    for start in range(0, len(moments), block_size):
        best = np.maximum(best, (queries @ moments[start:start + block_size].T).max(axis=1))
    return best
