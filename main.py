import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from data import FeatureStore, load_targets
from model import compile_video, score_video


INDEX_FORMAT = "tf-prvr-v2"


@contextmanager
def new_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb"):
        pass
    try:
        yield path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def build_index(args):
    with FeatureStore(args.videos, 2) as videos, new_output(args.index) as path:
        with h5py.File(path, "w") as index:
            index.attrs.update(format=INDEX_FORMAT, complete=False, wavelet=args.wavelet, beta=args.beta)
            index.attrs["backbone"] = videos.file.attrs.get("backbone", "unspecified")
            groups = index.create_group("videos")
            dimension = None
            for position, video_id in enumerate(videos.ids, 1):
                frames = videos[video_id]
                if dimension is None:
                    dimension = frames.shape[1]
                    index.attrs["dimension"] = dimension
                if frames.shape[1] != dimension:
                    raise ValueError(f"{video_id}: inconsistent video feature dimension")
                moments, spans, hierarchy = compile_video(frames, args.wavelet, args.beta)
                group = groups.create_group(video_id)
                group.create_dataset("features", data=moments.astype(np.float32))
                group.create_dataset("spans", data=spans)
                group.attrs.update(frames=len(frames), levels=len(hierarchy.offsets) - 1, nodes=len(hierarchy.features))
                if position % 100 == 0 or position == len(videos.ids):
                    print(f"Indexed {position}/{len(videos.ids)} videos", flush=True)
            index.attrs["complete"] = True


def retrieve(args):
    if args.batch_size < 1 or args.top_k < 1:
        raise ValueError("batch-size and top-k must be positive")
    with h5py.File(args.index, "r") as index, FeatureStore(args.queries, 1) as queries:
        if index.attrs.get("format") != INDEX_FORMAT or not index.attrs.get("complete", False):
            raise ValueError("Incomplete or unsupported video index; rebuild it with the index command")
        video_ids = sorted(index["videos"].keys())
        if not video_ids:
            raise ValueError("Video index is empty")
        dimension = int(index.attrs["dimension"])
        video_backbone = index.attrs.get("backbone", "unspecified")
        query_backbone = queries.file.attrs.get("backbone", "unspecified")
        if "unspecified" not in (video_backbone, query_backbone) and video_backbone != query_backbone:
            raise ValueError("Video and query features use different backbones")
        targets = load_targets(args.annotations, queries.ids, video_ids)
        cutoffs = (1, 5, 10, 100)
        hits = np.zeros(len(cutoffs), dtype=np.int64)
        with new_output(args.output) as path, path.open("w", encoding="utf-8") as stream:
            count = 0
            for query_ids, features in queries.query_batches(args.batch_size, dimension):
                scores = np.empty((len(query_ids), len(video_ids)), dtype=np.float32)
                for column, video_id in enumerate(video_ids):
                    moments = index["videos"][video_id]["features"][...]
                    scores[:, column] = score_video(features, moments)
                order = np.argsort(-scores, axis=1, kind="stable")
                for row, query_id in enumerate(query_ids):
                    ranked = order[row, :args.top_k]
                    result = {
                        "query_id": query_id,
                        "results": [{"video_id": video_ids[i], "score": float(scores[row, i])} for i in ranked],
                    }
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    if targets is not None:
                        for cutoff_index, cutoff in enumerate(cutoffs):
                            retrieved = {video_ids[i] for i in order[row, :cutoff]}
                            hits[cutoff_index] += bool(retrieved & targets[query_id])
                count += len(query_ids)
                print(f"Retrieved {count}/{len(queries.ids)} queries", flush=True)
        if targets is not None:
            metrics = {f"R@{cutoff}": float(100 * hit / count) for cutoff, hit in zip(cutoffs, hits)}
            metrics["SumR"] = sum(metrics.values())
            print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index")
    index.add_argument("--videos", required=True)
    index.add_argument("--index", required=True)
    index.add_argument("--wavelet", default="db4")
    index.add_argument("--beta", type=float, default=0.7)
    index.set_defaults(run=build_index)
    search = commands.add_parser("search")
    search.add_argument("--index", required=True)
    search.add_argument("--queries", required=True)
    search.add_argument("--output", required=True)
    search.add_argument("--annotations")
    search.add_argument("--batch-size", type=int, default=64)
    search.add_argument("--top-k", type=int, default=100)
    search.set_defaults(run=retrieve)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
