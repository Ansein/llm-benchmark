from pathlib import Path
from pypdf import PdfReader

out_dir = Path('_pdf_txt')
out_dir.mkdir(exist_ok=True)

pdfs = [
    '_ICDE_2026__llm_amplify_inequality_in_data_market (3).pdf',
    'Too Much Data Prices and Inefficiencies in Data Markets.pdf',
    'The Economics of Social Data.pdf',
    'Data-enabled learning, network effects, and competitive advantage.pdf',
    'To Partner or Not to Partner The Partnership Between Platforms and Data Brokers in Two-sided Markets.pdf',
    'Personalization and privacy choice.pdf',
]

for name in pdfs:
    pdf_path = Path(name)
    out_path = out_dir / (pdf_path.stem + '.txt')
    reader = PdfReader(str(pdf_path))

    parts = []
    total = len(reader.pages)
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ''
        except Exception:
            text = ''
        parts.append(f"\n\n=== PAGE {i+1} / {total} ===\n\n")
        parts.append(text)

    out_path.write_text(''.join(parts), encoding='utf-8', errors='ignore')
    print(f"extracted: {pdf_path.name} -> {out_path} (pages={total})")
