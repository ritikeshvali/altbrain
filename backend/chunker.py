"""structure-aware recursive chunking for RAG"""

from dataclasses import dataclass

SEPARATORS = ["\n## ", "\n# ", "\n### ", "\n\n", "\n", ". ", " "]

@dataclass
class Chunk:
    text: str
    index: int

def split_keep(text, separator):
    if separator == "":
        return list(text)
    parts = text.split(separator)
    out = []
    for i, p in enumerate(parts):
        out.append(p+separator if i<len(parts)-1 else p)
    return [p for p in out if p]

def recursive_split(text, max_chars, separators):
    sep, remaining = "", []
    for i, s in enumerate(separators):
        if s in text:
            sep, remaining = s, separators[i+1:]
            break

    out = []
    for part in split_keep(text, sep):
        if len(part) <= max_chars:
            out.append(part)
        elif remaining:
            out.extend(recursive_split(part, max_chars, remaining))
        else:
            for j in range(0, len(part), max_chars):
                out.append(part[j:j+max_chars])
    return out

def merge(splits, max_chars, overlap):
    chunks, cur = [], ""
    for s in splits:
        if cur and len(cur) + len(s)>max_chars:
            chunks.append(cur.strip())
            cur = cur[-overlap:] if overlap else ""
        cur = cur + s

    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def chunk_text(text, max_chars=1200, overlap=150):
    splits = recursive_split(text, max_chars, SEPARATORS)
    merged = merge(splits, max_chars, overlap)
    return [Chunk(t, i) for i, t in enumerate(merged)]