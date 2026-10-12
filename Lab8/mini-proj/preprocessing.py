import re

import numpy as np

# Character-level vocabulary. Index 0 = padding, 1 = unknown character.
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 |-&'.,()/%+:!"
CHAR2IDX = {c: i + 2 for i, c in enumerate(ALPHABET)}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
VOCAB_SIZE = len(ALPHABET) + 2


def clean_text(text):
    """Lowercase and collapse repeated whitespace."""

    return re.sub(r"\s+", " ", str(text).lower()).strip()


def encode_text(text, max_len=96):
    """Turn one string into a fixed-length vector of character ids."""

    ids = [CHAR2IDX.get(c, 1) for c in clean_text(text)[:max_len]]
    ids += [0] * (max_len - len(ids))          # right-pad with zeros
    return np.array(ids, dtype=np.int32)


def decode_text(ids):
    """Inverse of encode_text (used only to display samples)."""

    return "".join(IDX2CHAR.get(int(i), "?") for i in ids if i != 0)


def to_features(texts, max_len=96):

    return np.stack([encode_text(t, max_len) for t in texts]).astype(np.int32)