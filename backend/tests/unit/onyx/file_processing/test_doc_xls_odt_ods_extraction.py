"""Extraction for the legacy and OpenDocument formats: .doc (antiword),
.xls (xlrd), .odt and .ods (odfpy), plus the spreadsheet dispatchers that
route staging and previews by extension."""

import shutil
from io import BytesIO
from pathlib import Path
from typing import IO

import pytest
from odf import opendocument
from odf import table as odf_table
from odf import text as odf_text

from onyx.file_processing.extract_file_text import (
    count_odt_embedded_images,
    ods_to_text,
    read_odt_file,
    spreadsheet_sheet_extraction,
    stage_spreadsheet_sheets,
    xls_to_text,
)
from onyx.file_processing.file_types import OnyxFileExtensions, OnyxMimeTypes
from onyx.file_store.models import ChatFileType
from onyx.server.query_and_chat.chat_utils import mime_type_to_chat_file_type

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(name="sample_doc_bytes")
def sample_doc_bytes_fixture() -> bytes:
    return FIXTURES.joinpath("sample.doc").read_bytes()


@pytest.fixture(name="sample_xls_bytes")
def sample_xls_bytes_fixture() -> bytes:
    return FIXTURES.joinpath("sample.xls").read_bytes()


def _odt_bytes(with_image: bool = False) -> bytes:
    doc = opendocument.OpenDocumentText()
    doc.text.addElement(odf_text.H(outlinelevel=1, text="Ведомость выдачи"))
    doc.text.addElement(odf_text.P(text="Обычный абзац с текстом."))
    if with_image:
        # real-world ODTs store pictures under Pictures/
        doc.addPicture("Pictures/image1.png", "image/png", _tiny_png())
    table = odf_table.Table(name="Таблица1")
    for values in (("Товар", "Количество"), ("сгущенное молоко", "150")):
        row = odf_table.TableRow()
        for value in values:
            cell = odf_table.TableCell(valuetype="string")
            cell.addElement(odf_text.P(text=value))
            row.addElement(cell)
        table.addElement(row)
    doc.text.addElement(table)
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _ods_bytes(numeric: bool = False) -> bytes:
    doc = opendocument.OpenDocumentSpreadsheet()
    table = odf_table.Table(name="Лист1")
    for values in (("Товар", "Количество"), ("сгущенное молоко", "150")):
        row = odf_table.TableRow()
        for value in values:
            cell = odf_table.TableCell(valuetype="string")
            cell.addElement(odf_text.P(text=value))
            row.addElement(cell)
        table.addElement(row)
    if numeric:
        numeric_row = odf_table.TableRow()
        float_cell = odf_table.TableCell(valuetype="float", value="150.5")
        numeric_row.addElement(float_cell)
        int_cell = odf_table.TableCell(valuetype="float", value="42")
        numeric_row.addElement(int_cell)
        date_cell = odf_table.TableCell(valuetype="date", datevalue="2008-07-11")
        numeric_row.addElement(date_cell)
        table.addElement(numeric_row)
    doc.spreadsheet.addElement(table)
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def test_doc_extraction_handles_cyrillic_and_tables(
    sample_doc_bytes: bytes,
) -> None:
    pytest.importorskip("shutil", reason="stdlib")
    if shutil.which("antiword") is None:
        pytest.skip("antiword is not installed on this host")

    from onyx.file_processing.extract_file_text import doc_to_text

    text = doc_to_text(BytesIO(sample_doc_bytes), "sample.doc")

    assert "Ведомость выдачи" in text
    assert "Обычный абзац с текстом" in text
    assert "сгущенное молоко" in text


def test_doc_image_extraction_carves_embedded_rasters(
    sample_doc_bytes: bytes,
) -> None:
    # olefile is pure Python, so unlike the text test this runs everywhere
    from onyx.file_processing.extract_file_text import (
        count_doc_embedded_images,
        extract_doc_images,
    )

    images = list(extract_doc_images(BytesIO(sample_doc_bytes)))

    assert len(images) == 1
    blob, name = images[0]
    assert name.endswith(".png")
    assert blob.startswith(b"\x89PNG\r\n\x1a\n")

    with BytesIO(sample_doc_bytes) as f:
        assert count_doc_embedded_images(f, cap=5) == 1
    with BytesIO(sample_doc_bytes) as f:
        # over the cap the count short-circuits at cap+1
        assert count_doc_embedded_images(f, cap=0) == 1


def test_xls_extraction_produces_per_sheet_csv(sample_xls_bytes: bytes) -> None:
    text = xls_to_text(BytesIO(sample_xls_bytes), "sample.xls")

    assert "Товар,Количество" in text
    assert "сгущенное молоко,150" in text
    # integral numbers must not be rendered as floats
    assert "150.0" not in text


def test_xls_sheet_extraction_reports_sheet_titles(
    sample_xls_bytes: bytes,
) -> None:
    sheets = spreadsheet_sheet_extraction(BytesIO(sample_xls_bytes), "sample.xls")

    titles = [title for _csv_text, title in sheets]
    assert titles == ["Данные", "Empty"]


def _tiny_png() -> bytes:
    import struct
    import zlib

    def chunk(tag: bytes, payload: bytes) -> bytes:
        section = struct.pack(">I", len(payload)) + tag + payload
        return section + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 8 for _ in range(8))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_odt_extraction_renders_paragraphs_and_tables() -> None:
    text, images = read_odt_file(BytesIO(_odt_bytes()), "sample.odt")

    assert "Ведомость выдачи" in text
    assert "Обычный абзац с текстом." in text
    assert "Товар | Количество" in text
    assert "сгущенное молоко | 150" in text
    assert images == []


def test_odt_image_extraction_and_count() -> None:
    png = _tiny_png()
    document_bytes = _odt_bytes(with_image=True)

    images = list(
        read_odt_file(BytesIO(document_bytes), "sample.odt", extract_images=True)[1]
    )
    assert len(images) == 1
    assert images[0][0] == png

    with BytesIO(document_bytes) as f:
        assert count_odt_embedded_images(f, cap=5) == 1
    with BytesIO(document_bytes) as f:
        assert count_odt_embedded_images(f, cap=0) == 1


def test_ods_extraction_produces_per_sheet_csv() -> None:
    text = ods_to_text(BytesIO(_ods_bytes()), "sample.ods")

    assert "Товар,Количество" in text
    assert "сгущенное молоко,150" in text


def test_ods_numeric_cells_use_office_values_not_display_text() -> None:
    text = ods_to_text(BytesIO(_ods_bytes(numeric=True)), "sample.ods")

    assert "150.5,42,2008-07-11" in text


def test_stage_spreadsheet_sheets_stages_csv_per_sheet(
    sample_xls_bytes: bytes,
) -> None:
    staged: list[tuple[str, str]] = []

    def stage(data: IO[bytes], content_type: str) -> str:
        assert content_type == "text/csv"
        file_id = f"staged-{len(staged)}"
        staged.append((file_id, data.read().decode("utf-8")))
        return file_id

    sheets = stage_spreadsheet_sheets(
        BytesIO(sample_xls_bytes), stage, file_name="sample.xls"
    )

    # the fully-empty sheet is skipped, matching the xlsx staging behavior
    assert [sheet.title for sheet in sheets] == ["Данные"]
    assert all(sheet.csv_file_id.startswith("staged-") for sheet in sheets)
    assert "сгущенное молоко,150" in staged[0][1]


def test_new_extensions_are_allowed() -> None:
    for extension in (".doc", ".odt", ".xls", ".ods"):
        assert extension in OnyxFileExtensions.ALL_ALLOWED_EXTENSIONS

    assert {".xls", ".ods"} <= OnyxFileExtensions.SPREADSHEET_EXTENSIONS
    assert {".doc", ".odt"} <= OnyxFileExtensions.DOCUMENT_EXTENSIONS


def test_new_mimes_classify_into_the_right_chat_file_types() -> None:
    assert mime_type_to_chat_file_type("application/msword") == ChatFileType.DOC
    assert (
        mime_type_to_chat_file_type("application/vnd.oasis.opendocument.text")
        == ChatFileType.DOC
    )
    assert (
        mime_type_to_chat_file_type("application/vnd.ms-excel") == ChatFileType.TABULAR
    )
    assert (
        mime_type_to_chat_file_type("application/vnd.oasis.opendocument.spreadsheet")
        == ChatFileType.TABULAR
    )
    assert "application/vnd.ms-excel" in OnyxMimeTypes.TABULAR_MIME_TYPES
