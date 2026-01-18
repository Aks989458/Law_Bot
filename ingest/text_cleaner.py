import re
from pathlib import Path

def clean_text(text):
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)       # page numbers
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)     # broken words
    text = re.sub(r"\n(?=[a-z])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    headers = [
        "THE INDIAN PENAL CODE",
        "CODE OF CRIMINAL PROCEDURE",
        "CONSTITUTION OF INDIA",
        "INDIAN EVIDENCE ACT"
    ]
    for h in headers:
        text = text.replace(h, "")

    text = re.sub(r"\n\s*(Section|Article)\s+", r"\n\1 ", text)
    return text.strip()

def clean_all():
    input_dir = Path("/content/law-hybrid-rag/data/raw_text")
    output_dir = Path("/content/law-hybrid-rag/data/cleaned_text")
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob("*.txt"):
        raw = file.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw)
        (output_dir / file.name).write_text(cleaned, encoding="utf-8")

if __name__ == "__main__":
    clean_all()
