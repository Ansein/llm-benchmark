"""
PDF reading and section extraction utilities.

Dependencies: pdfplumber (pip install pdfplumber)
Falls back to pypdf if pdfplumber is unavailable.
"""

import re
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend: try pdfplumber first, fall back to pypdf
# ---------------------------------------------------------------------------

try:
    import pdfplumber
    _BACKEND = "pdfplumber"
except ImportError:
    pdfplumber = None
    try:
        from pypdf import PdfReader as _PdfReader
        _BACKEND = "pypdf"
    except ImportError:
        _PdfReader = None
        _BACKEND = "none"
        logger.warning("No PDF backend found. Install pdfplumber: pip install pdfplumber")

try:
    import pytesseract
    from PIL import Image, ImageOps
    _OCR_AVAILABLE = True
except ImportError:
    pytesseract = None
    Image = None
    ImageOps = None
    _OCR_AVAILABLE = False

_MIN_TEXT_CHARS_PER_PAGE = 120
_MIN_PRINTABLE_RATIO = 0.85


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    return printable / max(len(text), 1)


def _quality_score(text: str) -> float:
    if not text:
        return 0.0
    return len(text.strip()) * _printable_ratio(text)


def _evaluate_page_quality(text: str) -> dict[str, Any]:
    stripped = text.strip()
    char_count = len(stripped)
    p_ratio = _printable_ratio(text)
    garbled_markers = ("�", "\x00")
    garbled_count = sum(text.count(m) for m in garbled_markers)
    low_quality = char_count < _MIN_TEXT_CHARS_PER_PAGE or p_ratio < _MIN_PRINTABLE_RATIO
    return {
        "char_count": char_count,
        "printable_ratio": round(p_ratio, 4),
        "garbled_marker_count": garbled_count,
        "low_quality": low_quality,
        "quality_score": round(_quality_score(text), 2),
    }


def _ocr_page_pdfplumber(page) -> str:
    if not _OCR_AVAILABLE:
        return ""
    try:
        # Render page image and run OCR as fallback for low-quality text pages.
        pil = page.to_image(resolution=250).original
        if pil is None:
            return ""
        if pil.mode != "L":
            pil = pil.convert("L")
        pil = ImageOps.autocontrast(pil)
        text = pytesseract.image_to_string(pil)
        return text or ""
    except Exception as e:
        logger.warning(f"OCR failed on page: {e}")
        return ""


def _extract_pages_with_report(pdf_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract page texts and return (pages, quality_report)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    page_results: list[dict[str, Any]] = []

    if _BACKEND == "pdfplumber":
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                extracted = page.extract_text() or ""
                quality = _evaluate_page_quality(extracted)
                method = "text"
                selected_text = extracted

                used_ocr = False
                if quality["low_quality"] and _OCR_AVAILABLE:
                    ocr_text = _ocr_page_pdfplumber(page)
                    ocr_quality = _evaluate_page_quality(ocr_text)
                    if ocr_quality["quality_score"] > quality["quality_score"]:
                        selected_text = ocr_text
                        quality = ocr_quality
                        method = "ocr_fallback"
                        used_ocr = True

                page_results.append(
                    {
                        "page_index": i,
                        "method": method,
                        "used_ocr": used_ocr,
                        "quality": quality,
                        "text": selected_text,
                    }
                )

            return page_results, _build_quality_report(pdf_path, page_results)

    if _BACKEND == "pypdf":
        reader = _PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            quality = _evaluate_page_quality(extracted)
            page_results.append(
                {
                    "page_index": i,
                    "method": "text",
                    "used_ocr": False,
                    "quality": quality,
                    "text": extracted,
                }
            )
        return page_results, _build_quality_report(pdf_path, page_results)

    raise RuntimeError("No PDF backend available. pip install pdfplumber")


# ---------------------------------------------------------------------------
# Section detection heuristics
# ---------------------------------------------------------------------------

# Common academic paper section headers with confidence scores.
_SECTION_PATTERNS = [
    ("abstract", r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)?[\.\)]?\s*abstract\s*$", 1.0),
    ("introduction", r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)?[\.\)]?\s*introduction\s*$", 1.0),
    ("related_work", r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)?[\.\)]?\s*related\s+work\s*$", 0.95),
    ("model_setup", r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)?[\.\)]?\s*(?:model|model\s+setup|setup)\s*$", 0.95),
    ("equilibrium_definition", r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)?[\.\)]?\s*equilibrium(?:\s+definition)?\s*$", 0.95),
    ("main_results", r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)?[\.\)]?\s*(?:main\s+)?results?\s*$", 0.9),
    ("comparative_statics", r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)?[\.\)]?\s*comparative\s+statics?\s*$", 0.95),
    ("welfare", r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)?[\.\)]?\s*welfare\s*$", 0.9),
    ("numerical_examples", r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)?[\.\)]?\s*(?:numerical\s+)?examples?\s*$", 0.95),
    ("conclusion", r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)?[\.\)]?\s*conclusion\s*$", 0.95),
    ("appendix", r"^\s*(?:appendix|app\.?)\s*[A-Z]?\s*$", 0.95),
    ("references", r"^\s*references?\s*$", 1.0),
]

_COMPILED = [(name, re.compile(p, re.IGNORECASE), conf) for name, p, conf in _SECTION_PATTERNS]


def _match_section_header(line: str) -> tuple[str | None, float]:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None, 0.0
    for canonical, pat, conf in _COMPILED:
        if pat.match(stripped):
            return canonical, conf

    # Generic numbered heading: "2.3 Pricing Rule", "IV. Robustness"
    if re.match(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)[\.\)]\s+[A-Za-z][A-Za-z0-9 ,:\-]{2,70}$", stripped):
        key = re.sub(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)[\.\)]\s+", "", stripped).strip().lower()
        key = re.sub(r"\s+", "_", key)
        return key, 0.75

    # All-caps short headings often appear in PDFs.
    if stripped.isupper() and 3 <= len(stripped.split()) <= 8:
        key = re.sub(r"\s+", "_", stripped.lower())
        return key, 0.65

    return None, 0.0


def _split_into_sections(full_text: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """
    Heuristically split text into sections by detected headers.
    Returns ({section_name: section_text}, header_metadata).
    """
    lines = full_text.split("\n")
    sections: dict[str, list[str]] = {}
    header_meta: list[dict[str, Any]] = []
    current_section = "preamble"
    sections[current_section] = []

    for line_no, line in enumerate(lines, start=1):
        section_key, conf = _match_section_header(line)
        if section_key:
            current_section = section_key
            sections[current_section] = []
            header_meta.append(
                {
                    "line": line_no,
                    "header": line.strip(),
                    "normalized": current_section,
                    "confidence": conf,
                }
            )
        else:
            sections[current_section].append(line)

    return (
        {k: "\n".join(v).strip() for k, v in sections.items() if v},
        header_meta,
    )


def _build_quality_report(pdf_path: Path, page_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_pages = len(page_results)
    ocr_pages = sum(1 for p in page_results if p["used_ocr"])
    low_quality_pages = sum(1 for p in page_results if p["quality"]["low_quality"])
    avg_printable_ratio = (
        sum(p["quality"]["printable_ratio"] for p in page_results) / max(total_pages, 1)
    )

    report = {
        "pdf_path": str(pdf_path),
        "backend": _BACKEND,
        "ocr_available": _OCR_AVAILABLE,
        "summary": {
            "total_pages": total_pages,
            "ocr_fallback_pages": ocr_pages,
            "low_quality_pages": low_quality_pages,
            "avg_printable_ratio": round(avg_printable_ratio, 4),
        },
        "pages": [
            {
                "page_index": p["page_index"],
                "method": p["method"],
                "used_ocr": p["used_ocr"],
                **p["quality"],
            }
            for p in page_results
        ],
    }
    return report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_full_text(pdf_path: str | Path) -> str:
    """Return the full text of the PDF as a string."""
    full_text, _ = extract_full_text_with_report(pdf_path)
    return full_text


def extract_full_text_with_report(pdf_path: str | Path) -> tuple[str, dict[str, Any]]:
    """
    Return:
      - full_text: merged page text (text extraction + OCR fallback)
      - quality_report: page-level extraction quality report
    """
    pages, report = _extract_pages_with_report(pdf_path)
    full_text = "\n".join((p["text"] or "").strip() for p in pages if (p["text"] or "").strip())
    return full_text, report


def extract_sections_from_text(full_text: str) -> dict[str, str]:
    """Split an existing full-text string into normalized sections."""
    raw_sections, header_meta = _split_into_sections(full_text)

    structured: dict[str, str] = {"full_text": full_text}
    structured.update(raw_sections)
    structured["_section_detection"] = {
        "headers_detected": len(header_meta),
        "headers": header_meta,
    }
    return structured


def extract_sections(pdf_path: str | Path) -> dict[str, str]:
    """
    Extract and structure the paper by sections.

    Returns a dict with canonical keys where detected, plus raw text:
    {
        "full_text": "...",
        "abstract": "...",
        "model_setup": "...",
        "equilibrium": "...",
        "comparative_statics": "...",
        "numerical_examples": "...",
        "appendix": "...",
        ...  (other detected sections)
    }
    """
    full_text, _ = extract_full_text_with_report(pdf_path)
    return extract_sections_from_text(full_text)


def search_paper(query: str, full_text: str, top_k: int = 5, window: int = 300) -> list[str]:
    """
    Simple keyword-based search within paper text.
    Returns up to top_k passages (±window chars around each hit).

    For richer semantic search, plug in an embedding model here later.
    """
    query_words = [w.lower() for w in re.split(r"\W+", query) if len(w) > 2]
    if not query_words:
        return []

    # Score each position by how many query words appear nearby
    text_lower = full_text.lower()
    hits: list[tuple[int, int]] = []  # (score, position)

    step = 100
    for i in range(0, len(full_text) - window, step):
        chunk = text_lower[i : i + window]
        score = sum(1 for w in query_words if w in chunk)
        if score > 0:
            hits.append((score, i))

    hits.sort(key=lambda x: -x[0])

    # Deduplicate overlapping windows and extract passages
    passages: list[str] = []
    seen_positions: set[int] = set()
    for _, pos in hits:
        if any(abs(pos - p) < window for p in seen_positions):
            continue
        seen_positions.add(pos)
        start = max(0, pos - 50)
        end = min(len(full_text), pos + window)
        passages.append(full_text[start:end].strip())
        if len(passages) >= top_k:
            break

    return passages
