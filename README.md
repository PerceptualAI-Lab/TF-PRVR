<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS-2026-635BFF?style=flat-square" alt="NeurIPS 2026">
</p>

<h1 align="center">TF-PRVR:<br>Training-Free Partially Relevant Video Retrieval</h1>

<p align="center">
  <a href="https://giyeolkim.github.io/">Giyeol Kim</a> &nbsp;&nbsp;&nbsp;
  <a href="https://sites.google.com/view/pai-lab/members_1/faculty?authuser=0">Chanho Eom</a><sup>&dagger;</sup>
</p>

<p align="center">
  <a href="https://sites.google.com/view/pai-lab/home?authuser=0"><b>Perceptual AI Lab</b></a><br>
  GSAIM, Chung-Ang University
</p>

<p align="center">
  <sup>&dagger;</sup>Corresponding author
</p>

<p align="center">
  <a href="https://perceptualai-lab.github.io/TF-PRVR/"><img src="assets/links/project-page.svg" height="36" alt="Project Page"></a>
  &nbsp;
  <a href="https://perceptualai-lab.github.io/TF-PRVR/#demo"><img src="assets/links/demo.svg" height="36" alt="Demo"></a>
  &nbsp;
  <a href="https://arxiv.org/pdf/2610.07925"><img src="assets/links/paper.svg" height="36" alt="Paper"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2610.07925"><img src="assets/links/arxiv.svg" height="36" alt="arXiv"></a>
</p>

<p align="center">
  TF-PRVR retrieves untrimmed videos from natural-language queries using frozen vision-language features, adaptive temporal segments, and multi-scale relevance propagation&mdash;without task-specific training.
</p>

---

## Overview

Partially relevant video retrieval asks whether a video contains a moment matching a query. TF-PRVR adapts the temporal representation to each video and combines evidence from the same moment across scales.

<p align="center">
  <a href="assets/framework.png"><img src="assets/framework.png" width="100%" alt="TF-PRVR framework: frozen features, wavelet-based hierarchical segmentation, multi-scale graph propagation, and moment-aware scoring"></a>
</p>

1. **Video-specific hierarchical segmentation.** Changes in the direction of frame-feature evolution form a temporal semantic signal. Wavelet detail responses identify boundaries at multiple scales, yielding segments with coherent visual semantics.
2. **Graph-based relevance propagation.** A query-independent graph links segments through within-scale semantic similarity and temporal proximity, and through temporal overlap across adjacent scales.
3. **Moment-aware retrieval.** Query relevance is propagated through the graph, averaged across scales at each temporal location, and maximized over time to produce the video score.

**Frozen backbones · No PRVR-specific optimization · Query-independent video preprocessing**

## Results

### Generality across frozen backbones

TF-PRVR improves over the corresponding zero-shot baseline across all three evaluated backbones on **ActivityNet Captions** (Table 7a).

| Backbone | Zero-shot SumR | TF-PRVR SumR | Gain |
| :--- | ---: | ---: | ---: |
| CLIP-L/14 | 172.9 | **193.9** | +21.0 |
| SigLIP-L/14 | 178.7 | **198.5** | +19.8 |
| EVA-CLIP-L/14 | 180.4 | **200.8** | +20.4 |

### Benchmark comparison

TF-PRVR is evaluated alongside training-based approaches on **ActivityNet Captions**, **Charades-STA**, and **TVR** without PRVR-specific training. The training-based methods are evaluated in-domain.

<p align="center">
  <a href="assets/main-results.png"><img src="assets/main-results.png" width="100%" alt="Paper Table 6: in-domain training-based and training-free retrieval results on ActivityNet Captions, Charades-STA, and TVR"></a>
</p>

## Quick start

### 1. Install

```bash
git clone https://github.com/PerceptualAI-Lab/TF-PRVR.git
cd TF-PRVR
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.10 or newer is required. The feature-level retrieval pipeline runs on CPU with NumPy, SciPy, PyWavelets, and h5py.

### 2. Prepare pre-extracted features

This release provides the core feature-level retrieval pipeline. Supply features extracted by a frozen image–text encoder pair, such as **EVA-CLIP-L/14**.

| File | HDF5 dataset key | Array shape | Content |
| :--- | :--- | :--- | :--- |
| `videos.h5` | Video ID | `[T, D]` | Chronologically ordered, uniformly sampled frame embeddings |
| `queries.h5` | Query ID | `[D]` | Projected EOS embedding in the same shared space |

Feature dimensions are inferred. Frame sequences may have different lengths and must not contain padding. Use root-level HDF5 datasets with IDs that do not contain `/`. Both modalities must use the same encoder pair and projection space. An optional root attribute, `backbone`, can identify that pair in both files.

For evaluation, prepare one JSONL record per query with all relevant video IDs:

```json
{"query_id": "query_0001", "video_ids": ["video_0042"]}
{"query_id": "query_0002", "video_ids": ["video_0017", "video_0023"]}
```

### 3. Build the video index

```bash
python main.py index \
  --videos data/videos.h5 \
  --index outputs/index.h5
```

### 4. Retrieve and evaluate

```bash
python main.py search \
  --index outputs/index.h5 \
  --queries data/queries.h5 \
  --annotations data/queries.jsonl \
  --output outputs/rankings.jsonl \
  --batch-size 64 \
  --top-k 100
```

Omit `--annotations` for retrieval only. Rankings are saved as JSONL, and evaluation prints R@1, R@5, R@10, R@100, and SumR. Recall is computed against the full candidate set independently of the number of saved results. Ties follow lexicographic video ID order. Existing output files are preserved; choose a new output path for another run.

## Implementation

| File | Role |
| :--- | :--- |
| [`model.py`](model.py) | Semantic signal, wavelet segmentation, graph construction, propagation, and scoring |
| [`data.py`](data.py) | HDF5 feature loading and explicit query-to-video annotations |
| [`requirements.txt`](requirements.txt) | Runtime and research-environment dependencies |

The code is currently being polished and will be released soon.

## Citation

```bibtex
@misc{kim2026tfprvr,
  title={TF-PRVR: Training-Free Partially Relevant Video Retrieval},
  author={Giyeol Kim and Chanho Eom},
  year={2026},
  eprint={2610.07925},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.07925}
}
```
