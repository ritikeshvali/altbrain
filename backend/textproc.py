""" Turn an uploaded file in clean text, then into overlapping chunks. """

import io
import re
from pathlib import Path
from typing import List

from pypdf import PdfReader

def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    if suffix in [".txt", ".md"]:
        return data.decode("utf-8", errors="replace")

    raise ValueError(f"Unsupported file type: {suffix}")

def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00","")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """"
    Cut text into pieces of at most `size` characters.
    
    Each cut is moved back to the nearest paragraph, line or sentence end so
    a chunk does not stop mid-sentence. The next chunk starts `overlap`
    characters before the cut, so a fact that sits on the border is in both.
    """
    if overlap >= size // 2:
        raise ValueError("overlap must be less than half of size")

    text = clean_text(text=text)
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start+size, len(text))
        if end < len(text):
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep)
                if cut > size//2:
                    end = start + cut + len(sep)
                    break

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        start = end - overlap
    return chunks