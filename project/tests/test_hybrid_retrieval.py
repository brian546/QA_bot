from project.backend.app.services.hybrid_retrieval import reciprocal_rank_fusion


def test_hybrid_fusion_deduplicates_by_chunk_id() -> None:
    lexical = [
        {"chunk_id": "c1", "filename": "a.pdf", "page": 1, "text": "foo"},
        {"chunk_id": "c2", "filename": "a.pdf", "page": 2, "text": "bar"},
    ]
    semantic = [
        {"chunk_id": "c2", "filename": "a.pdf", "page": 2, "text": "bar"},
        {"chunk_id": "c3", "filename": "b.pdf", "page": 1, "text": "baz"},
    ]
    fused = reciprocal_rank_fusion(lexical, semantic, lexical_weight=1.0, semantic_weight=1.0, top_k=5)
    ids = [row["chunk_id"] for row in fused]
    assert len(ids) == len(set(ids))
    assert "c2" in ids
