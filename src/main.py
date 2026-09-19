from pypdf import PdfReader

def load_file(filepath):
    reader = PdfReader(filepath)
    pages = []

    # extract text from all pages in specified PDF
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        # if text it not null and no trailing/leading spaces, add to pages list
        if text and text.strip():
            pages.append({
                "text": text,
                "source": filepath,
                "page": page_num,
            })

    # print(pages)
    return pages

load_file("data/raw_docs/ALLIANZ_20251219.pdf")