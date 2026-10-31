import io

import pytest
from pypdf import PdfWriter

from textproc import chunk_text, clean_text, extract_text


def test_chunks_respect_size_and_end_at_sentences():
    text = ("Sentence number one is here. " * 200).strip()
    chunks = chunk_text(text, size=500, overlap=100)
    assert all(len(c) <= 500 for c in chunks)
    assert all(c.endswith(".") for c in chunks[:-1])


def test_neighbouring_chunks_overlap():
    text = ("Sentence number one is here. " * 200).strip()
    a, b = chunk_text(text, size=500, overlap=100)[:2]
    assert a[-20:] in b


def test_short_text_is_one_chunk_not_a_chunk_plus_a_sliver():
    # v1 returned [1204 chars, 4 chars] for this text: a useless duplicate tail
    text = " ".join(f"Item {i} is described here." for i in range(45))
    assert len(chunk_text(text, size=1500, overlap=300)) == 1


def test_empty_and_whitespace_give_no_chunks():
    assert chunk_text("", 500, 100) == []
    assert chunk_text("  \n\n  ", 500, 100) == []


def test_overlap_must_be_smaller_than_half_the_size():
    with pytest.raises(ValueError):
        chunk_text("x. " * 100, size=100, overlap=60)


def test_text_files_and_unsupported_types():
    assert extract_text("a.md", "héllo".encode()) == "héllo"
    with pytest.raises(ValueError):
        extract_text("a.exe", b"x")


def test_pdf_page_without_text_does_not_crash():
    buf = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(buf)
    assert extract_text("blank.pdf", buf.getvalue()).strip() == ""


def test_clean_text_collapses_whitespace():
    assert clean_text("a  b\r\n\n\n\nc\x00") == "a b\n\nc"