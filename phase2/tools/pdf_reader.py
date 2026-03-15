"""
PDF reading and section extraction utilities.

Dependencies: pdfplumber (pip install pdfplumber)
Falls back to pypdf if pdfplumber is unavailable.
"""

import re
import logging
from pathlib import Path
from typing import Optional

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


def _extract_raw_text(pdf_path: str | Path) -> str:
    """Extract all text from a PDF as a single string."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if _BACKEND == "pdfplumber":
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n".join(pages)

    if _BACKEND == "pypdf":
        reader = _PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    raise RuntimeError("No PDF backend available. pip install pdfplumber")


# ---------------------------------------------------------------------------
# Section detection heuristics
# ---------------------------------------------------------------------------

# Common academic paper section headers
_SECTION_PATTERNS = [
    r"^\s*(?:\d+\.?\s+)?abstract\s*$",
    r"^\s*(?:\d+\.?\s+)?introduction\s*$",
    r"^\s*(?:\d+\.?\s+)?model\s*$",
    r"^\s*(?:\d+\.?\s+)?(?:model\s+)?setup\s*$",
    r"^\s*(?:\d+\.?\s+)?equilibrium\s*$",
    r"^\s*(?:\d+\.?\s+)?(?:main\s+)?results?\s*$",
    r"^\s*(?:\d+\.?\s+)?comparative\s+statics?\s*$",
    r"^\s*(?:\d+\.?\s+)?welfare\s*$",
    r"^\s*(?:\d+\.?\s+)?(?:numerical\s+)?examples?\s*$",
    r"^\s*(?:\d+\.?\s+)?conclusion\s*$",
    r"^\s*(?:appendix|app\.?)\s*",
    r"^\s*references?\s*$",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _SECTION_PATTERNS]


def _is_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    return any(pat.match(stripped) for pat in _COMPILED)


def _split_into_sections(full_text: str) -> dict[str, str]:
    """
    Heuristically split text into sections by detected headers.
    Returns {section_name: section_text}.
    """
    lines = full_text.split("\n")
    sections: dict[str, list[str]] = {}
    current_section = "preamble"
    sections[current_section] = []

    for line in lines:
        if _is_section_header(line):
            current_section = line.strip().lower()
            sections[current_section] = []
        else:
            sections[current_section].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_full_text(pdf_path: str | Path) -> str:
    """Return the full text of the PDF as a string."""
    return _extract_raw_text(pdf_path)


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
    full_text = _extract_raw_text(pdf_path)
    raw_sections = _split_into_sections(full_text)

    # Normalize keys to snake_case canonical names
    canonical_map = {
        "model": "model_setup",
        "model setup": "model_setup",
        "setup": "model_setup",
        "equilibrium": "equilibrium_definition",
        "comparative statics": "comparative_statics",
        "comparative static": "comparative_statics",
        "results": "main_results",
        "main results": "main_results",
        "numerical example": "numerical_examples",
        "numerical examples": "numerical_examples",
        "example": "numerical_examples",
    }

    structured: dict[str, str] = {"full_text": full_text}
    for raw_key, text in raw_sections.items():
        canonical = canonical_map.get(raw_key, raw_key.replace(" ", "_"))
        structured[canonical] = text

    return structured


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
