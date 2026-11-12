"""hybrid retieval: dense and BM25, with RRF(reciprocal rank fusion) and cross-encoder rerank"""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def dense_rank(query_emb, chunk_embs):
    q = np.asarray(query_emb, dtype=np.float32)
    m = np.asarray(chunk_embs, dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-8
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-8
    scores = m@q
    return list(np.argsort(scores)[::-1])

def bm25_rank(query, texts):
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    return list(np.argsort(scores)[::-1])

def rrf(rankings, k=60):
    fused = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0/(k+pos)
    return sorted(fused, key=fused.get, reverse=True)

def retrieve(query, query_emb, chunks, chunk_embs, top_k=5, fuse_n=20):
    texts = [c["text"] for c in chunks]
    dense = dense_rank(query_emb, chunk_embs)
    sparse = bm25_rank(query, texts)
    fused = rrf([dense, sparse])[:fuse_n]

    pairs = [(query, texts[i]) for i in fused]
    rerank_scores = get_reranker().predict(pairs)
    order = np.argsort(rerank_scores)[::-1]
    return [chunks[fused[i]] for i in order[:top_k]]

def load_corpus(coll):
    data = coll.get(include=["embeddings", "documents", "metadatas"])
    metas = data.get("metadatas") or [{}] * len(data["documents"])
    chunks = [{"text": d, "meta": m or {}} for d, m in zip(data["documents"], metas)]
    return chunks, data["embeddings"]


def embed_query(embedding_function, query):
    return embedding_function([query])[0]