import io

from openpyxl import Workbook

from project.backend.app.services.parser import parse_document_pages


def test_parse_xlsx_uses_one_page_per_non_empty_sheet() -> None:
    workbook = Workbook()
    first = workbook.active
    first.title = "Summary"
    first.append(["metric", "value"])
    first.append(["revenue", 125000])

    second = workbook.create_sheet("Details")
    second.append(["id", "owner"])
    second.append(["42", "ops"])

    payload = io.BytesIO()
    workbook.save(payload)
    workbook.close()

    pages = parse_document_pages(payload.getvalue(), "report.xlsx")

    assert len(pages) == 2
    assert pages[0]["filename"] == "report.xlsx"
    assert pages[0]["page"] == 1
    assert "Sheet: Summary" in str(pages[0]["text"])
    assert "revenue | 125000" in str(pages[0]["text"])

    assert pages[1]["filename"] == "report.xlsx"
    assert pages[1]["page"] == 2
    assert "Sheet: Details" in str(pages[1]["text"])
    assert "42 | ops" in str(pages[1]["text"])