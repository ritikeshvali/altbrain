""" Test doubles shared by several test files without network and API keys """
import hashlib
import numpy as np
from chromadb.api.types import EmbeddingFunction

class WordHashEmbedder(EmbeddingFunction):
    """Fake embedder: texts that share words get similar vectors."""

    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for text in input:
            v = np.zeros(64, dtype=np.float32)
            for word in text.lower().split():
                v[int(hashlib.md5(word.strip(".,?").encode()).hexdigest(), 16)%64] += 1
            out.append(v/(np.linalg.norm(v) or 1))
        return out

    @staticmethod
    def name():
        return "word-hash-test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return WordHashEmbedder()