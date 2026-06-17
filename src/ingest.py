import fitz

pdf_path = "data/sample.pdf"

document = fitz.open(pdf_path)
pages = []

for page_num in range(len(document)):
    page = document[page_num]

    text = page.get_text()

    pages.append(
        {
            "page": page_num + 1,
            "text": text
        }
    )
print(f"Total pages: {len(pages)}")

print("\nFirst page preview:\n")

print(pages[0]["text"][:500])

print("\nPage 100 preview:\n")
print(pages[99]["text"][:500])