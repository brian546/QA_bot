from project.backend.app.services.lexical_retrieval import build_bm25_index, retrieve_lexical


def test_lexical_retrieval_finds_exact_term() -> None:
    chunks = [
        {"chunk_id": "a", "filename": "doc1.pdf", "page": 1, "text": "Policy ID ZX-778 applies."},
        {"chunk_id": "b", "filename": "doc2.pdf", "page": 2, "text": "General summary content."},
    ]
    index, _ = build_bm25_index(chunks)
    results = retrieve_lexical("ZX-778", chunks, index, top_k=2)
    assert results
    assert results[0]["chunk_id"] == "a"
