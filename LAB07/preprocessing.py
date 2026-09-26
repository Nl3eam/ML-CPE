import json
import re
from collections import Counter

import numpy as np

# Reserved ids: 0 = padding, 1 = unknown word
PAD_ID = 0
OOV_ID = 1

_CLEAN = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")


def preprocess_text(text, min_words=1):
    """Clean one review string. None if unusable.

    Text version of preprocess_image(): lowercase instead of BGR->RGB,
    punctuation stripping instead of resizing.
    """

    if text is None:
        return None

    text = str(text).lower()

    # Digits carry little meaning here ("noodle 5 pack"), keep them as <num>
    text = _CLEAN.sub(" ", text)
    text = re.sub(r"\b\d+\b", " num ", text)
    text = _SPACES.sub(" ", text).strip()

    if len(text.split()) < min_words:
        return None

    return text


def build_vocabulary(texts, max_words=10000, min_freq=1):
    """Map the most common words to integer ids. Fit on TRAIN ONLY."""

    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocabulary = {}
    for word, count in counter.most_common():
        if count < min_freq or len(vocabulary) + 2 >= max_words:
            break
        vocabulary[word] = len(vocabulary) + 2   # 0 and 1 are reserved

    print(f"Vocabulary size: {len(vocabulary) + 2} "
          f"(unique words seen: {len(counter)})")

    return vocabulary


def texts_to_sequences(texts, vocabulary, max_len=24):
    """Turn each review into a fixed-length id sequence (post-padded)."""

    sequences = np.full((len(texts), max_len), PAD_ID, dtype=np.int32)

    for row, text in enumerate(texts):
        ids = [vocabulary.get(word, OOV_ID) for word in text.split()][:max_len]
        sequences[row, :len(ids)] = ids

    return sequences


def to_features(sequences):

    return np.ascontiguousarray(sequences, dtype=np.int32)


def save_vocabulary(vocabulary, path):
    with open(path, "w") as f:
        json.dump(vocabulary, f)


def load_vocabulary(path):
    with open(path) as f:
        return json.load(f)