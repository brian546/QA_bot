from project.backend.app.services.dedupe import normalize_file_key


def test_normalize_file_key() -> None:
    assert normalize_file_key("  Quarterly Report .PDF  ") == "quarterly report.pdf"
    assert normalize_file_key("A/B/C.pdf") == "c.pdf"
