import numpy as np


def normalize(values, power=1.0):
    values = np.asarray(values, dtype=np.float64)
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)

    return values / np.maximum(lengths, 1e-12) ** power


def text_features(tokens, mix):
    eos = []
    previous = []

    for caption in tokens:
        if len(caption) == 0:
            raise ValueError("A caption must contain at least one token.")

        eos.append(caption[-1])
        previous.append(caption[-2] if len(caption) > 1 else caption[-1])

    eos = normalize(np.stack(eos))
    previous = normalize(np.stack(previous))

    return normalize((1 - mix) * eos + mix * previous)


def correct_queries(features, caption_mean, strength):
    return normalize(features - strength * caption_mean)
