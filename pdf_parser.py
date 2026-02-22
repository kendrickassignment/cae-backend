"""
Corporate Accountability Engine (CAE) — Forensic PDF Parser
Built for Act For Farmed Animals (AFFA) / Sinergia Animal International

Extracts text from PDF files with page number preservation,
special handling for footnotes, tables, and appendices.
Uses PyMuPDF (fitz) — completely free, no API needed.
"""

import fitz  # PyMuPDF
import re
from dataclasses import dataclass, field


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    page_number: int
    full_text: str
    footnotes: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    section_type: str = "body"  # body, appendix, footnote, table_of_contents


@dataclass
class ParsedDocument:
    """Represents a fully parsed PDF document."""
    file_name: str
    page_count: int
    pages: list[PageContent]
    full_text_with_markers: str
    metadata: dict

    def get_total_chars(self) -> int:
        return len(self.full_text_with_markers)

    def get_total_words(self) -> int:
        return len(self.full_text_with_markers.split())


# ============================================================================
# FOOTNOTE DETECTION PATTERNS
# ============================================================================

FOOTNOTE_PATTERNS = [
    r'^\s*\d+\s+[A-Z]',           # "1 This refers to..."
    r'^\s*\*+\s',                   # "* Note that..."  "** Excluding..."
    r'^\s*†',                       # "† See appendix..."
    r'^\s*‡',                       # "‡ Data not available..."
    r'^\s*Note:\s',                 # "Note: This excludes..."
    r'^\s*Source:\s',               # "Source: Company data..."
    r'^\s*\([a-z]\)\s',            # "(a) Where supply..."
    r'^\s*\(i+\)\s',               # "(i) Excluding..."
    r'^\s*\(?\d+\)?\s*Where\s',    # "(1) Where feasible..."
    r'^\s*\*\s*Where\s',           # "* Where supply is..."
    r'^\s*\*\s*Exclu',             # "* Excluding..."
    r'^\s*\*\s*Subject\s',         # "* Subject to..."
]

# Keywords that indicate we're in an appendix or special section
APPENDIX_KEYWORDS = [
    "appendix", "annex", "supplementary", "additional data",
    "methodology note", "glossary", "definitions",
    "endnotes", "end notes", "references"
]

# Keywords indicating a section has regional/geographic data
GEOGRAPHIC_SECTION_KEYWORDS = [
    "regional", "by region", "by country", "by market",
    "geographic", "country-level", "market breakdown",
    "asia", "southeast asia", "indonesia", "apac",
    "global operations", "international operations"
]


def detect_footnotes(text: str) -> list[str]:
    """Extract lines that appear to be footnotes from a text block."""
    footnotes = []
    lines = text.split('\n')

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        for pattern in FOOTNOTE_PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE):
                footnotes.append(stripped)
                break

    return footnotes


def detect_section_type(text: str) -> str:
    """Determine if a page is body text, appendix, TOC, etc."""
    text_lower = text.lower()

    # Check for table of contents
    if "table of contents" in text_lower or "contents" == text_lower.strip():
        return "table_of_contents"

    # Check for appendix/annex sections
    for keyword in APPENDIX_KEYWORDS:
        if keyword in text_lower:
            return "appendix"

    return "body"


def extract_tables_as_markdown(page: fitz.Page) -> list[str]:
    """
    Extract tables from a PDF page and convert to markdown format.
    This helps solve the 'Table Decay' problem mentioned in the whitepaper.
    """
    tables = []

    try:
        # PyMuPDF 1.23+ has built-in table detection
        tab_finder = page.find_tables()

        if tab_finder and tab_finder.tables:
            for table in tab_finder.tables:
                try:
                    # Extract table data as list of lists
                    data = table.extract()

                    if not data or len(data) < 2:
                        continue

                    # Convert to markdown table
                    md_lines = []

                    # Header row
                    headers = [str(cell) if cell else "" for cell in data[0]]
                    md_lines.append("| " + " | ".join(headers) + " |")
                    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

                    # Data rows
                    for row in data[1:]:
                        cells = [str(cell) if cell else "" for cell in row]
                        # Pad if needed
                        while len(cells) < len(headers):
                            cells.append("")
                        md_lines.append("| " + " | ".join(cells[:len(headers)]) + " |")

                    tables.append("\n".join(md_lines))

                except Exception:
                    continue

    except AttributeError:
        # Older PyMuPDF version — skip table extraction
        pass

    return tables


def parse_pdf(file_path: str) -> ParsedDocument:
    """
    Parse a PDF file and extract all text with page markers,
    footnote detection, table extraction, and section classification.

    Args:
        file_path: Path to the PDF file

    Returns:
        ParsedDocument with all extracted content
    """
    doc = fitz.open(file_path)

    pages: list[PageContent] = []
    full_text_parts: list[str] = []
    metadata = {
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "subject": doc.metadata.get("subject", ""),
        "creator": doc.metadata.get("creator", ""),
        "total_pages": len(doc),
        "has_geographic_sections": False,
        "geographic_section_pages": [],
        "appendix_start_page": None,
        "footnote_pages": [],
    }

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_number = page_num + 1  # 1-indexed for human readability

        # Extract full text from page
        text = page.get_text("text")

        if not text or not text.strip():
            continue

        # Detect footnotes on this page
        footnotes = detect_footnotes(text)

        # Extract tables and convert to markdown
        tables = extract_tables_as_markdown(page)

        # Detect section type
        section_type = detect_section_type(text)

        # Track metadata
        if section_type == "appendix" and metadata["appendix_start_page"] is None:
            metadata["appendix_start_page"] = page_number

        if footnotes:
            metadata["footnote_pages"].append(page_number)

        # Check for geographic content
        text_lower = text.lower()
        for keyword in GEOGRAPHIC_SECTION_KEYWORDS:
            if keyword in text_lower:
                metadata["has_geographic_sections"] = True
                if page_number not in metadata["geographic_section_pages"]:
                    metadata["geographic_section_pages"].append(page_number)
                break

        # Build page content
        page_content = PageContent(
            page_number=page_number,
            full_text=text,
            footnotes=footnotes,
            tables=tables,
            section_type=section_type
        )
        pages.append(page_content)

        # Build text with page markers for the LLM
        page_marker = f"\n\n--- [PAGE {page_number}] ---\n\n"
        full_text_parts.append(page_marker)
        full_text_parts.append(text)

        # Append footnotes with special markers
        if footnotes:
            full_text_parts.append(f"\n[FOOTNOTES ON PAGE {page_number}]:\n")
            for fn in footnotes:
                full_text_parts.append(f"  FOOTNOTE: {fn}\n")

        # Append tables as markdown
        if tables:
            full_text_parts.append(f"\n[TABLES ON PAGE {page_number}]:\n")
            for i, table in enumerate(tables):
                full_text_parts.append(f"  TABLE {i+1}:\n{table}\n")

    doc.close()

    return ParsedDocument(
        file_name=file_path.split("/")[-1],
        page_count=len(doc) if hasattr(doc, '__len__') else metadata["total_pages"],
        pages=pages,
        full_text_with_markers="".join(full_text_parts),
        metadata=metadata
    )


def chunk_document(
    parsed_doc: ParsedDocument,
    chunk_size: int = 4000,
    overlap: int = 200
) -> list[dict]:
    """
    Split the parsed document into chunks for processing.
    Each chunk preserves its page number context.

    For smaller documents (< chunk_size), returns the full text as one chunk.
    For larger documents, splits with overlap to preserve context.

    Args:
        parsed_doc: The parsed PDF document
        chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks

    Returns:
        List of chunk dicts with text, start_page, end_page
    """
    full_text = parsed_doc.full_text_with_markers

    # If document fits in one chunk, return it directly
    if len(full_text) <= chunk_size:
        return [{
            "text": full_text,
            "start_page": 1,
            "end_page": parsed_doc.page_count,
            "chunk_index": 0
        }]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(full_text):
        end = start + chunk_size

        # Try to break at a page marker to keep pages together
        if end < len(full_text):
            page_marker_pos = full_text.rfind("--- [PAGE", start, end)
            if page_marker_pos > start:
                end = page_marker_pos

        chunk_text = full_text[start:end]

        # Find page numbers in this chunk
        page_numbers = re.findall(r'\[PAGE (\d+)\]', chunk_text)
        page_numbers = [int(p) for p in page_numbers]

        chunks.append({
            "text": chunk_text,
            "start_page": min(page_numbers) if page_numbers else 0,
            "end_page": max(page_numbers) if page_numbers else 0,
            "chunk_index": chunk_index
        })

        start = end - overlap
        chunk_index += 1

    return chunks


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        print("Example: python pdf_parser.py report.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Parsing: {pdf_path}")

    result = parse_pdf(pdf_path)

    print(f"\n{'='*60}")
    print(f"FILE: {result.file_name}")
    print(f"PAGES: {result.page_count}")
    print(f"TOTAL CHARACTERS: {result.get_total_chars():,}")
    print(f"TOTAL WORDS: {result.get_total_words():,}")
    print(f"{'='*60}")

    print(f"\nMETADATA:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")

    print(f"\nFOOTNOTE PAGES: {result.metadata['footnote_pages']}")
    print(f"GEOGRAPHIC SECTIONS: {result.metadata['geographic_section_pages']}")
    print(f"APPENDIX STARTS: Page {result.metadata['appendix_start_page']}")

    # Show first 500 chars of formatted text
    print(f"\nFIRST 500 CHARS OF FORMATTED TEXT:")
    print(result.full_text_with_markers[:500])
