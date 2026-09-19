from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# LOAD PDF
def load_file(filepath):
    """
    Step 1: Load PDF

    Extract text from all pages in specified PDF and returns as a list with the
    basic page schema: {text: str, source: str, page: int}.
    """

    reader = PdfReader(filepath)
    pages = []

    # extract text from all pages in specified PDF
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "text": text,
                "source": filepath,
                "page": page_num,
            })

    # print(pages)
    return pages

# CHUNKING
def chunk_pages(pages, chunk_size, overlap):
    """
    Step 2: Chunk/split the loaded pages

    Split the chunks by natural boundaries such as double newline (paragraphs), 
    single newlines, sentences, spaces, characters.

    chunk_size is the number of tokens/words in each chunk. 500 is a good number 
    as it is large enough to contain a complete cause or condition, but also 
    small enough to capture specific meaning.

    100 ≈ 1-2 sentences of overlap, enough to bridge splits and prevent 
    losing info at chunk boundaries.

    Each chunk inherits the schema from its parent page (i.e "text", "source", "page").
    """

    # RecursiveCharacterTextSplitter splits text at natural boundaries (i.e. separators)
    separators = ["\n\n", "\n", ". ", " ", ""]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, separators=separators,)
    chunks = []

    # use splitter to split loaded text into chunks using the defined separators
    for page in pages:
        # print(page)
        splits = splitter.split_text(page["text"])

        for i, chunked_text in enumerate(splits):
            chunks.append({
                "text": chunked_text,
                "source": page["source"],
                "page": page["page"],
                "chunk": i,
            })

    return chunks

# EMBEDDING CHUNKS
def embed_chunks(chunks):
    """
    Step 3: Convert chunks' text into numerical vectors.

    Uses the embedding model nomic-embed-text to map text to a 768-dimensional vector.
    This allows semantically similar chunks to have similar vectors.
    Eg: "Is jet skiing covered?" and "recreational water sports coverage" have close
    vectors despite sharing no keywords. 

    nomic-embed-text is used because it is free and runs locally via Ollama.
    """

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    texts = [chunk["text"] for chunk in chunks]

    vectors = embeddings.embed_documents(texts)

    return vectors




pages = load_file("data/raw_docs/ALLIANZ_20251219.pdf")
chunks = chunk_pages(pages, 500, 100) 
# print(chunks)
embeddings = embed_chunks(chunks)
# print(embeddings)