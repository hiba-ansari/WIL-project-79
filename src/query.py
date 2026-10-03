import chromadb
from langchain_ollama import ChatOllama, OllamaEmbeddings
import json
import re
from dataclasses import dataclass

import yaml

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
    temperature: float = 0.1
    refused: bool = False # used for UI (app.py) if answer is a refusal

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
SYSTEM_PROMPT = config['domains']['travel_insurance']['system_prompt']
TOP_K = config['retrieval']['top_k']
DEFAULT_LLM_MODEL = config.get('llm', {}).get('model', 'llama3') # default generation model, overridable per call via ask(model=...)
EMBEDDING_MODEL = config.get('embeddings', {}).get('model', 'nomic-embed-text') # embedding model must match the one used at ingestion


# refusal detection
REFUSAL_PATTERNS = [
    "do not contain", "does not contain", "not contain any", "no information",
    "outside the scope", "outside the provided", "cannot answer", "can't answer",
    "unable to answer", "not able to answer", "cannot determine",
    "i cannot", "i'm unable", "sorry, i cannot",
    "contact your insurer", "contact us directly", "not covered by the provided",
    "i will only answer", "policy-related inquiries only", "unrelated to the policy",
    "i'm only allowed", "i am only allowed", "i'm only able", "i am only able",
    "i must say", "not related to the policy",
]

_SOURCE_NARRATION = re.compile(
    r"(source\s*\d+|\bpages?\b|\[source|the\s+excerpts?\b|the\s+policy\b|"
    r"the\s+documents?\b|the\s+provided\b|this\s+document\b)"
    r"[^.]*?"
    r"\b(do(?:es)?\s*not\s+(?:contain|provide|mention|state|include)|not\s+contain|no\s+information)",
    re.IGNORECASE,
)

_BOILERPLATE_REFUSAL = re.compile(
    r"the\s+excerpts?\s+do\s+not\s+contain\s+this\s+information\.?",
    re.IGNORECASE,
)


def is_refusal(answer: str):
    """Check if the response refuses to answer overall."""

    low = answer.lower()
    kept = []
    for sent in re.split(r"(?<=[.!?])\s+", low):
        if _SOURCE_NARRATION.search(sent):
            continue

        sent = _BOILERPLATE_REFUSAL.sub("", sent).strip()
        if sent:
            kept.append(sent)
    core = " ".join(kept).strip()
    if not core:
        return True
    return any(p in core for p in REFUSAL_PATTERNS)


def retrieve(question, db_path, collection_name, insurer_filter, top_k = TOP_K):
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
    query_vector = OllamaEmbeddings(model=EMBEDDING_MODEL).embed_query(question)

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

def build_prompt(question, sources, system_prompt):
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

def generate_response(messages, model=DEFAULT_LLM_MODEL, temperature=0.1):
    """
    Step 3: Send augmented prompt to LLM and generate answer.

    temperature=0.1 because higher values increase hallucination risk, which is undesireable
    in the context of legal documents. This value is tunable so that the eval pipeline can
    help decide the most optimal value.

    model defaults to config.yaml llm.model; the eval pipeline passes its own value so
    different generators (e.g. gemma2:2b) can be compared.

    ChatOllama is used instead of the raw Ollama API because it provides built-in support for
    chat messages.

    gemma2:2b got stuck into a repetition loop at t=0.1 and errored with "token repeat limit reached". 
    To address this, the repeat_penalty parameter is introduced.
    """

    penalties = [None, 1.15, 1.3, 1.5]
    last_err = None
    for attempt, penalty in enumerate(penalties, 1):
        kwargs = {} if penalty is None else {"repeat_penalty": penalty}
        llm = ChatOllama(model=model, temperature=temperature, **kwargs)
        try:
            response = llm.invoke(messages)
            if attempt > 1:
                print(f"  [generation ok on attempt {attempt}, repeat_penalty={penalty}]")
            return response.content
        except Exception as err:
            last_err = err
            print(f"  [generation attempt {attempt} failed: {err}"
                  + (f" - retrying with repeat_penalty={penalties[attempt]}" if attempt < len(penalties) else "]"))
    raise last_err

def ask(question, db_path, collection_name, insurer, top_k, temperature=0.1,
        model=DEFAULT_LLM_MODEL):
    """
    Run full RAG query pipeline. All 3 methods are called here.

    temperature defaults to 0.1; the eval pipeline passes its own value so
    sweeps can measure generation behaviour across temperatures.

    model defaults to config.yaml llm.model; the eval pipeline passes its own
    value to override the default so sweeps can compare different generator models. 
    The embedding model must remain the same as the one used for ingestion.
    """

    # initialise variables
    k = top_k
    emb_model = EMBEDDING_MODEL
    llm_model = model

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
    messages = build_prompt(question, sources, SYSTEM_PROMPT)
    print(f"  Propmt assembled ({len(messages)} messages)")
    print(f"  Assembled prompt: {messages}")

    # call method 3: generate_response()
    print(f"[3/3] Generating answer with {llm_model}...")
    answer = generate_response(messages, model=llm_model, temperature=temperature)
    print(f"  Answer generated ({len(answer)} chars)")

    # return RAGResult object for later formatting
    return RAGResult(
        question=question,
        answer=answer,
        insurer=insurer,
        sources=sources,
        model=llm_model,
        top_k=k,
        temperature=temperature,
        refused=is_refusal(answer) or not sources
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



if __name__ == "__main__":
    insurer = "BUDGET-DIRECT"
    question = "What is the per-item limit for a laptop on the Comprehensive plan?"
    db_path = "./data/vector_db/"
    collection_name = "Travel_Insurance"
    result = ask(question, db_path, collection_name, insurer)
    print_result(result)
