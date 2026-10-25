PRESENTATION_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)

SPREADSHEET_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
# Macro-enabled Excel workbooks (.xlsm) — parseable by openpyxl just like xlsx.
# Stored lowercase; chat MIME classification normalizes to lowercase before
# membership checks.
SPREADSHEET_MACRO_MIME_TYPE = "application/vnd.ms-excel.sheet.macroenabled.12"
SPREADSHEET_MIME_TYPES = {
    SPREADSHEET_MIME_TYPE,
    SPREADSHEET_MACRO_MIME_TYPE,
    # Legacy binary Excel (.xls) — parsed by xlrd
    "application/vnd.ms-excel",
    # OpenDocument spreadsheet (.ods) — parsed by odfpy
    "application/vnd.oasis.opendocument.spreadsheet",
}
WORD_PROCESSING_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
# Legacy binary Word (.doc) — text extracted via the antiword system package
MS_WORD_MIME_TYPE = "application/msword"
# OpenDocument text (.odt) — parsed by odfpy
ODT_MIME_TYPE = "application/vnd.oasis.opendocument.text"
PDF_MIME_TYPE = "application/pdf"
PLAIN_TEXT_MIME_TYPE = "text/plain"


class OnyxMimeTypes:
    IMAGE_MIME_TYPES = {"image/jpg", "image/jpeg", "image/png", "image/webp"}
    CSV_MIME_TYPES = {"text/csv"}
    TABULAR_MIME_TYPES = CSV_MIME_TYPES | SPREADSHEET_MIME_TYPES
    TEXT_MIME_TYPES = {
        PLAIN_TEXT_MIME_TYPE,
        "text/markdown",
        "text/x-markdown",
        "text/x-log",
        "text/x-config",
        "text/tab-separated-values",
        "application/json",
        "application/xml",
        "text/xml",
        "application/x-yaml",
        "application/yaml",
        "text/yaml",
        "text/x-yaml",
    }
    DOCUMENT_MIME_TYPES = {
        PDF_MIME_TYPE,
        WORD_PROCESSING_MIME_TYPE,
        MS_WORD_MIME_TYPE,
        ODT_MIME_TYPE,
        PRESENTATION_MIME_TYPE,
        "message/rfc822",
        "application/epub+zip",
    }
    # Popular audio/video types accepted as chat uploads. Stored media keeps
    # whatever MIME the client declared; these sets classify uploads and gate
    # the inline-serving allowlist counterparts in file_store/serving.py.
    AUDIO_MIME_TYPES = {
        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/ogg",
        "audio/webm",
        "audio/aac",
        "audio/mp4",
        "audio/x-m4a",
        "audio/flac",
        "audio/opus",
    }
    VIDEO_MIME_TYPES = {
        "video/mp4",
        "video/webm",
        "video/ogg",
        "video/quicktime",
        "video/x-matroska",
        "video/x-msvideo",
    }

    ALLOWED_MIME_TYPES = IMAGE_MIME_TYPES.union(
        TEXT_MIME_TYPES, DOCUMENT_MIME_TYPES, TABULAR_MIME_TYPES
    )

    EXCLUDED_IMAGE_TYPES = {
        "image/bmp",
        "image/tiff",
        "image/gif",
        "image/svg+xml",
        "image/avif",
    }


class OnyxFileExtensions:
    SPREADSHEET_EXTENSIONS = {
        ".xlsx",
        ".xlsm",
        # Legacy binary Excel (.xls) — parsed by xlrd
        ".xls",
        # OpenDocument spreadsheet (.ods) — parsed by odfpy
        ".ods",
    }
    TABULAR_EXTENSIONS = {
        ".csv",
        ".tsv",
    } | SPREADSHEET_EXTENSIONS
    PLAIN_TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".mdx",
        ".conf",
        ".log",
        ".json",
        ".csv",
        ".tsv",
        ".xml",
        ".yml",
        ".yaml",
        ".sql",
    }
    DOCUMENT_EXTENSIONS = {
        ".pdf",
        ".docx",
        # Legacy binary Word (.doc) — extracted via antiword
        ".doc",
        # OpenDocument text (.odt) — parsed by odfpy
        ".odt",
        ".pptx",
        ".eml",
        ".epub",
        ".html",
    } | SPREADSHEET_EXTENSIONS
    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }
    AUDIO_EXTENSIONS = {
        ".mp3",
        ".wav",
        ".ogg",
        ".m4a",
        ".aac",
        ".flac",
        ".opus",
        ".weba",
    }
    VIDEO_EXTENSIONS = {
        ".mp4",
        ".webm",
        ".mov",
        ".mkv",
        ".avi",
        ".mpg",
        ".mpeg",
    }

    TEXT_AND_DOCUMENT_EXTENSIONS = PLAIN_TEXT_EXTENSIONS.union(DOCUMENT_EXTENSIONS)

    ALL_ALLOWED_EXTENSIONS = TEXT_AND_DOCUMENT_EXTENSIONS.union(IMAGE_EXTENSIONS)
