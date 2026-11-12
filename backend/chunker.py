"""structure-aware recursive chunking for RAG"""

from dataclasses import dataclass

SEPARATORS = ["\n## ", "\n# ", "\n### ", "\n\n", "\n", ". ", " "]

@dataclass
class Chunk:
    text: str
    index: int

def recursive_split(text, max_chars, separators):
    sep, remaining = "", []
    for i, s in enumerate(separators):
        if s in text:
            sep, remaining = s, separators[i+1:]
            break
    splits = text.split(sep) if sep else list(text)

    out = []
    for part in splits:
        if len(part) <= max_chars:
            out.append(part)
        elif remaining:
            out.extend(recursive_split(part, max_chars, remaining))
        else:
            for j in range(0, len(part), max_chars):
                out.append(part[j:j+max_chars])
    return out

def merge(splits, max_chars, overlap, sep=""):
    chunks, cur = [], ""
    for s in splits:
        if cur and len(cur) + len(s) + len(sep)>max_chars:
            chunks.append(cur.strip())
            cur = cur[-overlap:] if overlap else ""
        cur = (cur + sep + s) if cur else s

    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def chunk_text(text, max_chars=1200, overlap=150):
    splits = recursive_split(text, max_chars, SEPARATORS)
    merged = merge(splits, max_chars, overlap)
    return [Chunk(t, i) for i, t in enumerate(merged)]