import pytest

from pka_core import MultiCollectionPKA
from tests.fakes import WordHashEmbedder


@pytest.fixture
def pka(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = MultiCollectionPKA("sk-fake", db_path=str(tmp_path / "pka.db"))
    p.embedding_function = WordHashEmbedder()
    return p


def add(pka, collection, text, source):
    col = pka.chroma_client.get_collection(collection, embedding_function=pka.embedding_function)
    col.add(documents=[text], metadatas=[{"source": source}], ids=[f"{collection}-{source}"])


def test_search_returns_matching_chunk(pka):
    # v1 raised AttributeError here: self.RELEVANCE_THRESHOLD does not exist
    add(pka, "complete", "The Kafka consumer timeout was raised to 45 seconds", "notes.md")
    results = pka.search_multi_collection("Kafka consumer timeout", ["complete"])
    assert results and results[0]["source"] == "notes.md"


def test_unrelated_question_returns_nothing(pka):
    # v1 fell back to "top results anyway", so the model always got sources to hallucinate from
    add(pka, "professional", "The Kafka consumer timeout was raised to 45 seconds", "notes.md")
    assert pka.search_multi_collection("pizza recipe oven", ["professional"]) == []