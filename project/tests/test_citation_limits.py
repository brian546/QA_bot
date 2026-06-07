import pytest

from project.backend.app.core.config import Settings
from project.backend.app.services.qa import answer_with_evidence


def test_settings_reject_non_positive_citations_max() -> None:
    with pytest.raises(ValueError, match="CITATIONS_MAX_K"):
        Settings(
            OPENROUTER_API_KEY="x",
            OPENROUTER_MODEL="openai/gpt-oss-120b:free",
            CITATIONS_MAX_K=0,
        )


def test_settings_derive_retrieval_and_default_citations_from_citations_max() -> None:
    settings = Settings(
        OPENROUTER_API_KEY="x",
        OPENROUTER_MODEL="openai/gpt-oss-120b:free",
        CITATIONS_MAX_K=13,
    )

    assert settings.retrieval_lexical_k == 13
    assert settings.retrieval_semantic_k == 13
    assert settings.citations_default_k == 13


def test_answer_with_evidence_respects_citation_limit() -> None:
    settings = Settings(
        OPENROUTER_API_KEY="x",
        OPENROUTER_MODEL="openai/gpt-oss-120b:free",
        CITATIONS_MAX_K=20,
    )
    fused_rows = [
        {
            "chunk_id": f"doc:1:{idx}",
            "filename": "doc.pdf",
            "page": 1,
            "section": idx,
            "text": f"evidence {idx}",
        }
        for idx in range(12)
    ]

    _, citations = answer_with_evidence(
        settings=settings,
        question="What does the document say?",
        compressed_context="[doc:1:0] evidence",
        fused_rows=fused_rows,
        llm_settings=settings.default_llm_settings(),
        citation_limit=9,
    )

    assert len(citations) == 9