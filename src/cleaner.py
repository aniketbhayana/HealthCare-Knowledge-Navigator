import fitz
import re

pdf_path = "data/sample.pdf"

document = fitz.open(pdf_path)

page = document[99]

text = page.get_text()

# Fix words split across lines
text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

# Convert remaining line breaks into spaces
text = text.replace("\n", " ")

# Remove repeated spaces
text = re.sub(r'\s+', ' ', text)

print(text[:1000])