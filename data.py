import json
from pathlib import Path

import h5py
import numpy as np

from model import normalize, validate_features


class FeatureStore:
    def __init__(self, path, ndim):
        self.file = h5py.File(path, "r")
        self.ndim = ndim
        self.ids = sorted(self.file.keys())
        if not self.ids:
            self.file.close()
            raise ValueError(f"{path}: feature file is empty")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()

    def __getitem__(self, identifier):
        dataset = self.file[identifier]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{identifier}: expected a dataset at the HDF5 root")
        return validate_features(dataset[...], self.ndim, identifier)

    def query_batches(self, batch_size, dimension):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self.ids), batch_size):
            ids = self.ids[start:start + batch_size]
            values = [self[identifier] for identifier in ids]
            if any(value.shape != (dimension,) for value in values):
                raise ValueError(f"Queries must be EOS embeddings with shape ({dimension},)")
            yield ids, normalize(np.stack(values)).astype(np.float32)


def load_targets(path, query_ids, video_ids):
    if path is None:
        return None
    targets = {}
    candidates = set(video_ids)
    queries = set(query_ids)
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id, positives = row["query_id"], row["video_ids"]
            if query_id in targets:
                raise ValueError(f"Duplicate query_id in annotations: {query_id}")
            if not isinstance(positives, list) or not positives or not all(isinstance(v, str) for v in positives):
                raise ValueError(f"Annotation line {line_number}: video_ids must be a nonempty string list")
            if query_id not in queries or not set(positives) <= candidates:
                raise ValueError(f"Annotation line {line_number}: unknown query or video ID")
            targets[query_id] = set(positives)
    missing = queries - targets.keys()
    if missing:
        raise ValueError(f"Missing annotations for {len(missing)} queries")
    return targets
