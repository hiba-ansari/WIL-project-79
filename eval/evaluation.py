import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from query import ask, load_config

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
DB_PATH = PROJECT_ROOT / "data" / "vector_db"
COLLECTION_NAME = "Travel_Insurance"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama" 


# Cheap metrics (no LLM)
REFUSAL_PATTERNS = [
    "do not contain", "does not contain", "not contain any", "no information",
    "outside the scope", "outside the provided", "cannot answer", "can't answer",
    "unable to answer", "not able to answer", "cannot determine",
    "i cannot", "i'm unable", "sorry, i cannot",
    "contact your insurer", "contact us directly", "not covered by the provided",
    "i will only answer", "policy-related inquiries only", "unrelated to the policy",
]


def is_refusal(answer: str):
    """Check if response contains a refusal phrase."""

    low = answer.lower()
    return any(p in low for p in REFUSAL_PATTERNS)


def extract_cited_pages(answer: str):
    """Extract page numbers from response."""

    pages = set()
    for match in re.finditer(r"pages?\s+(\d+)(?:\s*[-–]\s*(\d+))?", answer, re.IGNORECASE):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        pages.update(range(start, min(end, 55) + 1))
    return sorted(pages)


def custom_metrics(item: dict, run_entry: dict):
    """Calculate cheap metrics for a given test question."""

    answer = run_entry["answer"]
    category = item["question_type"]
    relevant_pages = item["relevant_pages"]
    refused = is_refusal(answer)

    if category == "out_of_knowledge_base":
        ookb_correct = 1.0 if refused else 0.0
    else:
        ookb_correct = 0.0 if refused else 1.0

    retrieved_pages = [s["page"] for s in run_entry["sources"]]
    cited_pages = extract_cited_pages(answer)

    if category == "out_of_knowledge_base":
        retrieval_hit = precision_at_k = attribution_precision = None
    else:
        rp = set(relevant_pages)
        retrieval_hit = 1.0 if rp & set(retrieved_pages) else 0.0
        precision_at_k = (
            len(rp & set(retrieved_pages)) / len(retrieved_pages) if retrieved_pages else 0.0
        )
        attribution_precision = (
            len(rp & set(cited_pages)) / len(cited_pages) if cited_pages else None
        )

    return {
        "refused": 1.0 if refused else 0.0,
        "ookb_correct": ookb_correct,
        "retrieval_hit_at_k": retrieval_hit,
        "retrieval_precision_at_k": precision_at_k,
        "cited_pages": cited_pages,
        "attribution_precision": attribution_precision,
    }


# calculate RAGAS metrics
RAGAS_METRICS = ["faithfulness", "answer_correctness", "context_precision"]

def build_ragas_metrics(model: str, embedding_model: str):
    """Construct the three RAGAS metrics using Ollama."""

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from ragas.metrics import collections as rc

    client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    llm = llm_factory(model, provider="openai", client=client)
    embeddings = RagasOpenAIEmbeddings(client=client, model=embedding_model)

    return {
        "faithfulness": rc.Faithfulness(llm=llm),
        "answer_correctness": rc.AnswerCorrectness(llm=llm, embeddings=embeddings),
        "context_precision": rc.ContextPrecision(llm=llm),
    }

async def score_ragas_one(metrics: dict, item: dict, entry: dict):
    """
    Compute all RAGAS metrics for a single test item. Metrics for refusal answers are 
    not meaningful and thus recorded as None.
    """
    contexts = [s["text"] for s in entry["sources"]]
    # refused = is_refusal(entry["answer"])
    ookb = item["question_type"] == "out_of_knowledge_base"
    scores = {}

    for name, m in metrics.items():
        if ookb and name in ("faithfulness", "context_precision", "answer_correctness"):
            scores[name] = None
            continue
        try:
            if name == "faithfulness":
                r = await m.ascore(
                    user_input=item["question"],
                    response=entry["answer"],
                    retrieved_contexts=contexts,
                )
            elif name == "answer_correctness":
                r = await m.ascore(
                    user_input=item["question"],
                    response=entry["answer"],
                    reference=item["golden_answer"],
                )
            elif name == "context_precision":
                r = await m.ascore(
                    user_input=item["question"],
                    reference=item["golden_answer"],
                    retrieved_contexts=contexts,
                )
            else:
                raise KeyError(name)
            scores[name] = r.value
        except Exception as e:
            print(f"      ! {name} failed: {type(e).__name__}: {e}")
            scores[name] = None

    return scores


# collect: run the RAG pipeline and score
def collect(config: dict, domain: str, test_set: list[dict], limit: int | None,
            ids: list[str] | None, resume_file: Path | None = None,
            cheap_only: bool = False):
    """Run test question against the RAG pipeline."""

    items = select_items(test_set, limit, ids)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if resume_file:
        run = json.loads(resume_file.read_text(encoding="utf-8"))
        done = {e["id"] for e in run["entries"]}
        items = [t for t in items if t["id"] not in done]
        run_file = resume_file
        print(f"Resuming {run_file.name}: {len(done)} done, {len(items)} remaining")
    else:
        run_file = RESULTS_DIR / f"run_collect_{domain}_{datetime.now():%Y%m%d_%H%M%S}.json"
        run = {
            "mode": "collect",
            "domain": domain,
            "created": datetime.now().isoformat(timespec="seconds"),
            "config_snapshot": {
                "top_k": config.get("retrieval", {}).get("top_k", 5),
                "llm": config.get("llm", {}),
                "embeddings": config.get("embeddings", {}),
            },
            "entries": [],
        }

    ragas_metrics = {}
    if not cheap_only:
        model = config.get("llm", {}).get("model", "llama3")
        emb_model = config.get("embeddings", {}).get("model", "nomic-embed-text")
        print(f"Building RAGAS evaluator (model={model}, base={OLLAMA_BASE_URL})")
        ragas_metrics = build_ragas_metrics(model, emb_model)

    total = len(items)
    for i, item in enumerate(items, 1):
        print(f"\n[{i}/{total}] {item['id']}: {item['question'][:60]}...")
        t0 = time.perf_counter()

        # call RAG pipeline here
        result = ask(
            question=item["question"],
            db_path=DB_PATH,
            collection_name=COLLECTION_NAME,
            insurer=item.get("insurer") if item["question_type"] != "out_of_knowledge_base" else None,
        )
        latency = time.perf_counter() - t0

        entry = {
            "id": item["id"],
            "question": item["question"],
            "question_type": item["question_type"],
            "answer": result.answer,
            "sources": result.sources,
            "latency_s": round(latency, 2),
        }

        # cheap metrics (always calculated)
        entry["scores"] = custom_metrics(item, entry)

        # RAGAS metrics (unless command contains '--cheap-only')
        if ragas_metrics:
            print(f"  scoring RAGAS (may take minutes)...")
            ragas_scores = asyncio.run(score_ragas_one(ragas_metrics, item, entry))
            entry["scores"].update(ragas_scores)

        run["entries"].append(entry)
        save_run(run_file, run)
        print(f"  [{i}/{total}] saved (latency {latency:.1f}s)")

    print(f"\nCollected {total} answers in {run_file}")
    return run_file


# add / append to report
def mean_or_none(vals: list) -> float | None:
    """
    Calculate mean scores for each test question type (i.e. factual, reasoning, ookb).
    Return None if there are no valid values instead of 0, which would be misleading.
    """

    vals = [v for v in vals if v is not None]
    return round(statistics.fmean(vals), 3) if vals else None


def aggregate(run: dict):
    out = {}
    cats = ["factual", "reasoning", "out_of_knowledge_base"]
    keys = ["faithfulness", "answer_correctness", "context_precision",
            "ookb_correct", "retrieval_hit_at_k", "retrieval_precision_at_k",
            "attribution_precision", "refused"]
    groups = {"overall": run["entries"],
              **{c: [e for e in run["entries"] if e["question_type"] == c] for c in cats}}
    for gname, entries in groups.items():
        agg = {k: mean_or_none([e.get("scores", {}).get(k) for e in entries]) for k in keys}
        lats = [e["latency_s"] for e in entries if "latency_s" in e]
        agg["latency_p50_s"] = round(statistics.median(lats), 1) if lats else None
        agg["n"] = len(entries)
        out[gname] = agg
    return out


def report(run_file: Path):
    """Format evaluation report."""

    run = json.loads(run_file.read_text(encoding="utf-8"))
    agg = aggregate(run)

    print(f"\n{'='*90}")
    print(f"EVALUATION REPORT — {run_file.name}")
    print(f"{'='*90}")
    header = (f"{'group':<22}{'n':>4}{'faithfulness':>14}{'ansCorrectness':>16}{'contextPrec':>14}"
              f"{'ookb_ok':>10}{'retrievalHit':>14}{'retrievalPrec':>15}{'attributionPrec':>17}{'refused':>9}{'lat50':>7}")
    print(header)
    print("-" * len(header))
    for g, a in agg.items():
        print(f"{g:<22}{a['n']:>4}"
              f"{fmt(a,'faithfulness'):>14}{fmt(a,'answer_correctness'):>16}{fmt(a,'context_precision'):>14}"
              f"{fmt(a,'ookb_correct'):>10}{fmt(a,'retrieval_hit_at_k'):>14}"
              f"{fmt(a,'retrieval_precision_at_k'):>15}{fmt(a,'attribution_precision'):>17}"
              f"{fmt(a,'refused'):>9}{fmt(a,'latency_p50_s'):>7}")

    report_path = run_file.with_name(run_file.stem + "_report.json")
    report_path.write_text(json.dumps({"rag": agg, "run_file": str(run_file.name)}, indent=2, default=str),
                           encoding="utf-8")
    print(f"\nSaved report in {report_path}")


# helper functions
def fmt(a, k):
    """Format decimal values as 2 decimal places or a “—” dash"""

    v = a.get(k)
    return f"{v:.2f}" if isinstance(v, (int, float)) else "—"


def load_test_set(path: Path):
    """Load JSON test QnA set."""

    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_arg(value: str):
    """
    Refer to the latest run file for convenience when requesting eval report in terminal,
    instead of inputting the full path.
    """

    if value == "latest":
        files = [f for f in sorted(RESULTS_DIR.glob("run_collect_*.json"))
                 if not f.name.endswith("_report.json")]
        if not files:
            raise FileNotFoundError(f"No run_collect_*.json run files in {RESULTS_DIR} yet")
        return files[-1]
    return Path(value)


def select_items(test_set: list[dict], limit: int | None, ids: list[str] | None):
    """Allow specifying a subset of test questions (by ID no.) to run."""

    if ids:
        wanted = set(ids)
        items = [t for t in test_set if t["id"] in wanted]
        missing = wanted - {t["id"] for t in items}
        if missing:
            raise ValueError(f"Unknown test ids: {sorted(missing)}")
        return items
    return test_set[:limit] if limit else test_set


def save_run(path: Path, run: dict):
    """Save as tmp run file."""

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# CLI
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="RAG Evaluation")
    parser.add_argument("--mode", required=True, choices=["collect", "report"])
    parser.add_argument("--domain", default="travel_insurance")
    parser.add_argument("--test-set", default=str(PROJECT_ROOT / "data" / "evaluation" / "qna_dataset.json"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--run-file", help="Run JSON or 'latest' (report)")
    parser.add_argument("--resume", default=None, help="Run file to continue (collect)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None, help="Comma-separated test ids")
    parser.add_argument("--cheap-only", action="store_true",
                    help="Skip RAGAS metrics (fast, no Ollama evaluator needed)")
    args = parser.parse_args()

    config = load_config(args.config)
    test_set = load_test_set(Path(args.test_set))
    ids = args.ids.split(",") if args.ids else None
    resume = Path(args.resume) if args.resume else None

    if args.mode == "collect":
        collect(config, args.domain, test_set, args.limit, ids, resume, args.cheap_only)
    elif args.mode == "report":
        if not args.run_file:
            parser.error("--run-file required (path or 'latest')")
        report(resolve_run_arg(args.run_file))


if __name__ == "__main__":
    main()