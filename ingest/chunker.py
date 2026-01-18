from pathlib import Path
import re
import json

BASE_DIR = Path(__file__).resolve().parents[1]

def smart_chunk(text, act_name, max_chars=1200):
    chunks = []
    buffer = ""

    # VERY FLEXIBLE section detection
    section_regex = re.compile(
        r"^\s*(Section\s+\d+|Article\s+\d+|\d+\s*[.\-])",
        re.IGNORECASE
    )

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    for line in lines:
        # New section detected
        if section_regex.match(line) and len(buffer) > 200:
            chunks.append({
                "text": buffer.strip(),
                "act": act_name
            })
            buffer = line
        else:
            buffer += " " + line

        # Fallback: chunk by size if no sections
        if len(buffer) >= max_chars:
            chunks.append({
                "text": buffer.strip(),
                "act": act_name
            })
            buffer = ""

    if buffer.strip():
        chunks.append({
            "text": buffer.strip(),
            "act": act_name
        })

    return chunks

def chunk_all():
    input_dir = BASE_DIR / "data" / "cleaned_text"
    output_dir = BASE_DIR / "data" / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    for file in input_dir.glob("*.txt"):
        text = file.read_text(encoding="utf-8", errors="ignore")

        if len(text.strip()) < 500:
            print(f"⚠️ Skipping empty or broken file: {file.name}")
            continue

        act_name = file.stem.upper()
        chunks = smart_chunk(text, act_name)
        all_chunks.extend(chunks)

        print(f"✔ {file.name}: {len(chunks)} chunks")

    with open(output_dir / "law_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ TOTAL CHUNKS CREATED: {len(all_chunks)}")

if __name__ == "__main__":
    chunk_all()
