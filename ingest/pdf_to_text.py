import fitz  # PyMuPDF
from pathlib import Path

def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def convert_all():
    input_dir = Path("/content/law-hybrid-rag/data/pdfs")
    output_dir = Path("/content/law-hybrid-rag/data/raw_text")
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf in input_dir.glob("*.pdf"):
        raw_text = pdf_to_text(pdf)
        out_file = output_dir / f"{pdf.stem}.txt"
        out_file.write_text(raw_text, encoding="utf-8")

if __name__ == "__main__":
    convert_all()
