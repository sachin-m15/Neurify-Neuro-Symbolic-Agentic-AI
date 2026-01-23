import re

def clean_page_text(text: str) -> str:
    text = text.replace("\r", "\n")
    # Fix hyphenated words
    text = re.sub(r"-\n([a-zA-Z])", r"\1", text)
    # Remove page numbers (common in PDFs)
    text = re.sub(r"\n\d+\n", "\n", text)
    # Remove headers/footers (rough pattern)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip lines that are all caps and short (likely headers)
        if len(line) < 50 and line.isupper() and not any(c.islower() for c in line):
            continue
        # Skip lines with many numbers (page numbers, etc.)
        if len(re.findall(r'\d', line)) > len(line) / 2:
            continue
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
