import numpy as np
import pywt
from scipy.signal import find_peaks

from features import normalize


def temporal_signal(frames):
    velocity = np.diff(frames, axis=0)
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    direction = velocity / (speed + 1e-8)

    cosine = np.sum(direction[:-1] * direction[1:], axis=1)
    signal = 1 - np.clip(cosine, -1, 1)

    stationary = speed[:, 0] <= 1e-8
    signal[stationary[:-1] & stationary[1:]] = 0

    return signal


def frequency_boundaries(signal, wavelet, threshold, peak_distance,
                         peak_distance_scale):
    wavelet = pywt.Wavelet(wavelet)
    depth = pywt.dwt_max_level(len(signal), wavelet.dec_len)
    boundaries = []

    if depth == 0:
        return boundaries

    coefficients = pywt.wavedec(signal, wavelet, mode="symmetric", level=depth)

    for level in range(1, depth + 1):
        isolated = [np.zeros_like(part) for part in coefficients]
        isolated[-level] = coefficients[-level]

        response = pywt.waverec(isolated, wavelet, mode="symmetric")
        response = np.abs(response[:len(signal)])

        distance = max(peak_distance, peak_distance_scale * 2 ** (level - 1), 1)
        peaks = find_peaks(response, distance=distance)[0]
        cutoff = response.mean() + threshold * response.std()
        peaks = peaks[response[peaks] > max(cutoff, 1e-14)]

        boundaries.append(np.unique(peaks + 2))

    shared = np.empty(0, dtype=int)

    for level in range(len(boundaries) - 1, -1, -1):
        shared = np.union1d(shared, boundaries[level])
        boundaries[level] = shared.copy()

    return boundaries


def make_segments(frames, wavelet, threshold, peak_distance,
                  peak_distance_scale=0, max_frames=None):
    frames = np.asarray(frames, dtype=np.float64)

    if frames.ndim != 2 or len(frames) == 0 or not np.isfinite(frames).all():
        raise ValueError("Expected a nonempty array of finite frame features.")

    frames = normalize(frames)

    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive.")

        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
            frames = frames[indices]

    count = len(frames)
    boundaries = []

    if count >= 3:
        boundaries = frequency_boundaries(
            temporal_signal(frames), wavelet, threshold,
            peak_distance, peak_distance_scale,
        )

    scales = [np.column_stack((np.arange(count), np.arange(1, count + 1)))]

    for cuts in boundaries:
        edges = np.unique(np.r_[0, cuts, count])
        scales.append(np.column_stack((edges[:-1], edges[1:])))

    cumulative = np.vstack((np.zeros((1, frames.shape[1])), np.cumsum(frames, axis=0)))
    nodes = []
    offsets = [0]

    for spans in scales:
        totals = cumulative[spans[:, 1]] - cumulative[spans[:, 0]]
        lengths = spans[:, 1] - spans[:, 0]
        nodes.append(normalize(totals / lengths[:, None]))
        offsets.append(offsets[-1] + len(spans))

    return np.concatenate(nodes), np.concatenate(scales), np.array(offsets)
