import chromadb
from langchain_ollama import ChatOllama, OllamaEmbeddings
import json
from dataclasses import dataclass

@dataclass
class RAGResult:
    """
    Structured result from RAG query.
    """

    question: str
    answer: str
    insurer: str
    sources: list[dict]
    model: str
    top_k: int

def retrieve(question, db_path, collection_name, insurer_filter, top_k = 5):
    """
    Step 1: Retrieve most relevant chunks for the query

    - Embed the user's question using same model used during ingestion (nomic-embed-text)
      to ensure vector and chunk vectors are comparable/ in the same space
    - Query ChromaDB for the k nearest neighbours by cosine similarity
    - Return the chunks' text and schema
    """

    # initialise persistent client
    client = chromadb.PersistentClient(path=db_path)

    # get current collection
    collection = client.get_collection(name=collection_name)

    # embed query using same embedding model used during ingestion
    query_vector = OllamaEmbeddings(model="nomic-embed-text").embed_query(question)

    # filter by chosen insurer
    where_clause = None
    if insurer_filter:
        where_clause = {"insurer": insurer_filter}

    # store most relevant chunks
    results = collection.query(query_embeddings=[query_vector], 
                               n_results=top_k,
                               where=where_clause,
                               include=["documents", "metadatas", "distances"],)

    # flatten nested results and format into a clean list
    sources = []
    for doc, meta, dist in zip(results["documents"][0], 
                               results["metadatas"][0], 
                               results["distances"][0],
                               ):
        sources.append({
            "text": doc,
            "source": meta["source"],
            "insurer": meta["insurer"],
            "doc_date": meta["doc_date"],
            "page": meta["page"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })

    # print results in json
    print(f"Sources:\n{json.dumps(results, indent=2)}")

    return sources

def build_prompt(sources, system_prompt):
    """
    Step 2: Construct what the LLM will recieve before responding

    - Send system message (i.e. instructions for how to behave/persona)
    - Send user message (i.e. question and retrieved chunks as context. The source labels
      are added with each chunk so the LLM can cite them.)

    Context is given alongside the user message so the LLM treats it as primary input.
    """

    context_parts = []

    for i, src in enumerate(sources, 1):
        label = f"[Source {i}: {src['source']}, page {src['page']}]"
        context_parts.append(f"{label}\n{src['text']}")

    context_block = "\n\n---\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": (
                f"Use ONLY the following policy document excerpts to answer the question.\n\n"
                f"## Retrieved Context\n\n{context_block}\n\n"
                f"## Question\n\n{question}"
            ),
        },
    ]

    return messages

def generate_response(messages, temperature=0.1):
    """
    Step 3: Send augmented prompt to LLM and generate answer.

    temperature=0.1 because higher values increase hallucination risk, which is undesireable
    in the context of legal documents. This value is tunable so that the eval pipeline can
    help decide the most optimal value.

    ChatOllama is used instead of the raw Ollama API because it provides built-in support for
    chat messages.
    """

    llm = ChatOllama(model="llama3", temperature=temperature)
    response = llm.invoke(messages)

    return response.content

def ask(question, db_path, collection_name, insurer, top_k=5):
    """
    Run full RAG query pipeline. All 3 methods are called here.
    """

    # initialise variables
    k = top_k
    emb_model = "nomic-embed-text"
    llm_model = "llama3"
    temperature = 0.1
    system_prompt = "You are a helpful travel insurance assistant. Answer based on the provided context."

    # run query
    print(f"\n=== RAG QUERY ===")
    print(f"  Question: {question}")
    print(f"Top-K: {k} | Model: {llm_model} | Temperature: {temperature}")

    # call method 1: retrieve()
    if insurer:
        print(f"\nInsurer: {insurer}")

    print(f"\n[1/3] Retrieving relevant chunks...")
    sources = retrieve(question, db_path, collection_name, insurer, k)
    distances = [f"{s['distance']:.3f}" for s in sources]
    print(f"  Found {len(sources)} chunks (distances: {distances}")

    # call method 2: build_prompt()
    print(f"[2/3] Building augmented prompt for LLM...")
    messages = build_prompt(sources, system_prompt)
    print(f"  Propmt assembled ({len(messages)} messages)")
    print(f"  Assembled prompt: {messages}")

    # call method 3: generate_response()
    print(f"[3/3] Generating answer with {llm_model}...")
    answer = generate_response(messages)
    print(f"  Answer generated ({len(answer)} chars)")

    # return RAGResult object for later formatting
    return RAGResult(
        question=question,
        answer=answer,
        insurer=insurer,
        sources=sources,
        model=llm_model,
        top_k=k
    )

def print_result(result: RAGResult):
    """
    Format RAG result here.
    """

    print(f"\n{'='*50}")
    print("\nRAG RESPONSE:")

    print(f"\n{'='*50}")
    print(f"\nQUESTION: {result.question}")

    print(f"\n{'='*50}")
    print(f"\nANSWER:\n{result.answer}")

    print(f"\n{'='*50}")
    print(f"\nTOP_K: {result.top_k}")

    print(f"\n{'='*50}")
    print(f"\nSOURCES: {json.dumps(result.sources, indent=2)}\n\n({len(result.sources)} chunks retrieved)")
    for i, src in enumerate(result.sources, 1):
        print(f"  {i}. {src['source']} (page {src['page']}, distance {src['distance']:.3f})")

    print(f"\n{'='*50}\n")



insurer = "BUDGET-DIRECT"
question = "What is the per-item limit for a laptop on the Comprehensive plan?"
db_path = "./data/vector_db/"
collection_name = "Travel_Insurance"
# retrieve(question, db_path, collection_name)
result = ask(question, db_path, collection_name, insurer)
print_result(result)