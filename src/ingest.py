from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb
import re
from pathlib import Path


# PARSE PDF FILENAMES
pattern = re.compile(r"^([A-Z0-9]+(?:-[A-Z0-9]+)*)_(\d{8})\.pdf$") # COMPANY-NAME_YYYYMMDD.pdf 
def parse_document_filename(filename):
    """
    Extract insurer name and document data from specified naming convention (in Planning.md).
    """

    match = pattern.match(filename)
    if not match:
        raise ValueError(f"Invalid filename: {filename}")

    return {
        "insurer": match.group(1),
        "doc_date": match.group(2),
    }

# LOAD PDF
def load_pdfs(raw_docs_dir):
    """
    Step 1: Load PDF

    Extract text from all pages in specified PDF and returns as a list with the
    basic page schema: {text: str, source: str, page: int}.
    """

    
    pages = []
    pdf_dir = Path(raw_docs_dir) # store directory where all PDFs are uploaded

    # check if directory exists
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {raw_docs_dir}")

    # check if PDF files in stored directory exist
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {raw_docs_dir}")

    # loop through each PDF file and extract their text
    for pdf_path in pdf_files:
        doc_meta = parse_document_filename(pdf_path.name)
        print(f"  Loading {pdf_path.name} (insurer={doc_meta['insurer']}, date={doc_meta['doc_date']})")

        # extract text from all pages in current PDF
        reader = PdfReader(str(pdf_path))

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                pages.append({
                    "text": text,
                    "source": pdf_path.name,
                    "insurer": doc_meta["insurer"],
                    "doc_date": doc_meta["doc_date"],
                    "page": page_num,
                })

    print(pages)
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
                "insurer": page["insurer"],
                "doc_date": page["doc_date"],
                "page": page["page"],
                "chunk_index": i,
            })

    return chunks

# EMBEDDING CHUNKS
def embed_chunks(chunks, batch_size=64):
    """
    Step 3: Convert chunks' text into numerical vectors.

    Uses the embedding model nomic-embed-text to map text to a 768-dimensional vector.
    This allows semantically similar chunks to have similar vectors.
    Eg: "Is jet skiing covered?" and "recreational water sports coverage" have close
    vectors despite sharing no keywords. 

    nomic-embed-text is used because it is free and runs locally via Ollama.
    """

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    all_vectors = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk["text"] for chunk in batch]
        vectors = embeddings.embed_documents(texts)
        all_vectors.extend(vectors)

    return all_vectors

# STORE EMBEDDINGS
def store_embeddings(chunks, embeddings, db_path, collection_name):
    """
    Step 4: Store embeddings in ChromaDB

    ChromaDB is used because: 
    - It is lightweight and stores vectors alongside their
    source text and schema/metadata.
    - Provides fast cosine-similarity search when queried.
    - Supports metadata filtering.

    For each chunk, its id, embedding, document and metadata are stored.
    """

    client = chromadb.PersistentClient(path=db_path)

    # delete any exisitng data to avoid duplicates on rerun
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # create collection for current Allianz PDS document
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Travel insurance PDS chunks"}
    )

    # create schema for DB entries
    import uuid 
    ids = [uuid.uuid4() for c in chunks] # randomise UUID as ID value
    documents = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        metadatas.append(
            {
                "source": c["source"],
                "insurer": c["insurer"],
                "doc_date": c["doc_date"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
            }
        )

    # populate DB
    for i, c in enumerate(chunks):
        collection.add(
            ids=str(ids[i]),
            embeddings=embeddings[i],
            documents=documents[i],
            metadatas=metadatas[i],
        )

    print(f"  Stored {len(ids)} chunks in collection '{collection_name}'")
    print(f"  Vector DB location: {db_path}")

    return collection

# INGEST
def ingest():
    """
    Run full ingestion pipeline for all raw PDFs.
    """

    print(f"\n{'='*50}")
    print("\nINGESTION PIPELINE:")
    
    print(f"\n{'='*50}")
    print("\n[1/4] Loading PDFS...")
    pages = load_pdfs("data/raw_docs/")

    print(f"\n{'='*50}")
    print("\n[2/4] Chunking text...")
    chunks = chunk_pages(pages, 500, 100) 

    print(f"\n{'='*50}")
    print("\n[3/4] Embedding chunks...")
    embeddings = embed_chunks(chunks)

    print(f"\n{'='*50}")
    print("\n[4/4] Storing in ChromaDB...")
    collection = store_embeddings(chunks, embeddings, "data/vector_db/", "Travel_Insurance")

    result = collection.get(include=["documents"])
    print(f"Total stored documents: {len(result["documents"])}")


ingest()