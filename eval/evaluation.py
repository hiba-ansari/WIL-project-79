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

config = load_config()
TOP_K = config['retrieval']['top_k']


# Cheap metrics (no LLM)
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


def extract_cited_pages(answer: str):
    """Extract page numbers from response."""

    pages = set()
    for match in re.finditer(r"pages?\s+(\d+)(?:\s*[-–]\s*(\d+))?", answer, re.IGNORECASE):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        pages.update(range(start, min(end, 55) + 1))
    return sorted(pages)


def custom_metrics(item: dict, run_entry: dict, has_retrieval: bool = True):
    """Calculate cheap metrics for a given test question.

    has_retrieval=False is used for baseline runs: no chunks were retrieved,
    so page-level retrieval metrics are N/A (None, not 0 — a zero would
    wrongly read as "retrieval failed" instead of "no retrieval by design").
    """

    answer = run_entry["answer"]
    category = item["question_type"]
    relevant_pages = item["relevant_pages"]
    refused = is_refusal(answer)

    if category == "out_of_knowledge_base":
        ookb_correct = 1.0 if refused else 0.0
    else:
        ookb_correct = 0.0 if refused else 1.0

    retrieved_pages = [s["page"] for s in run_entry["sources"]] if has_retrieval else []
    cited_pages = extract_cited_pages(answer)

    if category == "out_of_knowledge_base":
        retrieval_hit = precision_at_k = attribution_precision = None
    elif not has_retrieval:
        rp = set(relevant_pages)
        retrieval_hit = precision_at_k = None
        attribution_precision = (
            len(rp & set(cited_pages)) / len(cited_pages) if cited_pages else None
        )
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

# run   
def vanilla_answer(question: str, model: str):
    """Answer without retrieval."""

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=model, temperature=0.1)
    prompt = (
        "You are a helpful travel insurance assistant. Answer the user's question "
        "to the best of your ability.\n\nQuestion: " + question
    )
    return llm.invoke(prompt).content


def baseline(config: dict, domain: str, test_set: list[dict], limit: int | None, ids: list[str] | None, resume_file: Path | None = None, cheap_only: bool = False):
    """Run the vanilla-LLM control over the same test questions."""

    items = select_items(test_set, limit, ids)
    model = config.get("llm", {}).get("model", "llama3")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if resume_file:
        run = json.loads(resume_file.read_text(encoding="utf-8"))
        done = {e["id"] for e in run["entries"]}
        items = [t for t in items if t["id"] not in done]
        run_file = resume_file
        print(f"Resuming {run_file.name}: {len(done)} done, {len(items)} remaining")
    else:
        run_file = RESULTS_DIR / f"run_baseline_{domain}_{datetime.now():%Y%m%d_%H%M%S}.json"
        run = {
            "mode": "baseline",
            "domain": domain,
            "created": datetime.now().isoformat(timespec="seconds"),
            "config_snapshot": {"llm": {"model": model}, "retrieval": "none"},
            "entries": [],
        }

    ragas_metrics = {}
    if not cheap_only:
        emb_model = config.get("embeddings", {}).get("model", "nomic-embed-text")
        print(f"Building RAGAS evaluator (model={model}, base={OLLAMA_BASE_URL})")
        all_metrics = build_ragas_metrics(model, emb_model)
        ragas_metrics = {"answer_correctness": all_metrics["answer_correctness"]}

    total = len(items)
    for i, item in enumerate(items, 1):
        print(f"\n[{i}/{total}] baseline {item['id']}: {item['question'][:60]}...")
        t0 = time.perf_counter()
        answer = vanilla_answer(item["question"], model)
        latency = time.perf_counter() - t0
        entry = {
            "id": item["id"],
            "question": item["question"],
            "question_type": item["question_type"],
            "answer": answer,
            "sources": [],
            "latency_s": round(latency, 2),
        }
        entry["scores"] = custom_metrics(item, entry, has_retrieval=False)
        if ragas_metrics:
            print(f"  scoring RAGAS (may take minutes)...")
            entry["scores"].update(asyncio.run(score_ragas_one(ragas_metrics, item, entry)))
        run["entries"].append(entry)
        save_run(run_file, run)
        print(f"  [{i}/{total}] saved (latency {latency:.1f}s)")

    print(f"\nBaseline collected {total} answers in {run_file}")
    print(f"Next: python eval/evaluation.py --mode report --run-file {run_file.name}")
    return run_file

# collect: run the RAG pipeline and score
def collect(config: dict, domain: str, test_set: list[dict], limit: int | None,
            ids: list[str] | None, resume_file: Path | None = None,
            cheap_only: bool = False):
    """Run test question against the RAG pipeline."""

    top_k = config.get("retrieval", {}).get("top_k", TOP_K)
    items = select_items(test_set, limit, ids)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if resume_file:
        run = json.loads(resume_file.read_text(encoding="utf-8"))
        done = {e["id"] for e in run["entries"]}
        items = [t for t in items if t["id"] not in done]
        run_file = resume_file
        print(f"Resuming {run_file.name}: {len(done)} done, {len(items)} remaining")
    else:
        run_file = RESULTS_DIR / f"run_collect_{domain}_k_{top_k}_{datetime.now():%Y%m%d_%H%M%S}.json"
        run = {
            "mode": "collect",
            "domain": domain,
            "created": datetime.now().isoformat(timespec="seconds"),
            "config_snapshot": {
                "top_k": config.get("retrieval", {}).get("top_k", TOP_K),
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
            top_k=top_k,
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


def evaluate(config: dict, test_set: list[dict], run_file: Path, cheap_only: bool = False):
    """(Re-)score a saved run file without re-running the pipeline.

    Works for both collect and baseline runs. Cheap metrics are free, so they
    are always recomputed; RAGAS metrics are skipped with --cheap-only.
    """

    run = json.loads(run_file.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in test_set}
    has_retrieval = run["mode"] == "collect"

    for entry in run["entries"]:
        # merge instead of replace: keep any RAGAS scores already on the entry so a --cheap-only re-score doesn't wipe faithfulness/correctness/context.
        entry.setdefault("scores", {}).update(
            custom_metrics(by_id[entry["id"]], entry, has_retrieval)
        )
    save_run(run_file, run)

    if cheap_only:
        print(f"\nCheap metrics updated in {run_file}")
        return

    names = RAGAS_METRICS if has_retrieval else ["answer_correctness"]
    model = run["config_snapshot"].get("llm", {}).get("model", "llama3")
    emb_model = run["config_snapshot"].get("embeddings", {}).get("model", "nomic-embed-text")
    print(f"Building RAGAS evaluator (model={model}, base={OLLAMA_BASE_URL})")
    all_metrics = build_ragas_metrics(model, emb_model)
    metrics = {n: m for n, m in all_metrics.items() if n in names}

    total = len(run["entries"])
    for i, entry in enumerate(run["entries"], 1):
        print(f"[{i}/{total}] RAGAS {entry['id']}...")
        entry["scores"].update(
            asyncio.run(score_ragas_one(metrics, by_id[entry["id"]], entry))
        )
        save_run(run_file, run)

    print(f"\nMetrics updated in {run_file}")


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


def report(run_file: Path, baseline_file: Path | None = None):
    """Format evaluation report.

    - Run on a baseline file  -> standalone BASELINE report (title carries
      'baseline' so it is never confused with RAG reports).
    - Run on a collect file   -> RAG report with baseline stats appended for
      later comparison (explicit --baseline-file, else the latest baseline run).
    """

    run = json.loads(run_file.read_text(encoding="utf-8"))
    agg = aggregate(run)
    is_baseline_run = run.get("mode") == "baseline"

    base_agg = None
    base_run_name = None
    if not is_baseline_run:
        bf = baseline_file or find_latest_baseline()
        if bf and bf.exists():
            base_run_name = bf.name
            base_agg = aggregate(json.loads(bf.read_text(encoding="utf-8")))

    title = "BASELINE EVALUATION REPORT" if is_baseline_run else "EVALUATION REPORT"
    print(f"\n{'='*90}")
    print(f"{title} — {run_file.name}")
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

    if base_agg:
        print(f"\n{'='*90}")
        print(f"RAG vs VANILLA BASELINE (no retrieval) — {base_run_name}")
        print(f"{'='*90}")
        cmp_header = (f"{'group':<22}{'rag_correct':>12}{'base_correct':>13}"
                      f"{'rag_ookb_ok':>12}{'base_ookb_ok':>13}{'rag_refused':>12}{'base_refused':>13}")
        print(cmp_header)
        print("-" * len(cmp_header))
        for g, a in agg.items():
            b = base_agg.get(g, {})
            print(f"{g:<22}{fmt(a,'answer_correctness'):>12}{fmt(b,'answer_correctness'):>13}"
                  f"{fmt(a,'ookb_correct'):>12}{fmt(b,'ookb_correct'):>13}"
                  f"{fmt(a,'refused'):>12}{fmt(b,'refused'):>13}")

    payload = {("baseline" if is_baseline_run else "rag"): agg}
    if base_agg:
        payload["baseline"] = base_agg
        payload["baseline_file"] = base_run_name
    payload["run_file"] = str(run_file.name)
    report_path = run_file.with_name(run_file.stem + "_report.json")
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved report in {report_path}")


def summary(report_file: Path):
    """Format a saved report as a RAG vs baseline summary Excel sheet.

    Columns: Metric | Group | RAG | Baseline | Delta.
    Context-dependent metrics (faithfulness, context precision, retrieval)
    are left blank for the baseline — a context-less run has no definition
    for them, and setting as 0 would be misleading.
    """

    import pandas as pd

    data = json.loads(report_file.read_text(encoding="utf-8"))
    rag = data["rag"]
    base = data.get("baseline")
    if base is None:
        bf = find_latest_baseline()
        base = aggregate(json.loads(bf.read_text(encoding="utf-8"))) if bf else None

    rows = [
        ("Right behaviour for scope, out_of_knowledge_base", "out_of_knowledge_base", "ookb_correct"),
        ("Right behaviour, factual", "factual", "ookb_correct"),
        ("Right behaviour, reasoning", "reasoning", "ookb_correct"),
        ("Answer page retrieved", "overall", "retrieval_hit_at_k"),
        ("Answer page cited (attribution)", "overall", "attribution_precision"),
        ("Correctness", "overall", "answer_correctness"),
        ("Faithfulness", "overall", "faithfulness"),
        ("Context precision", "overall", "context_precision"),
        ("Refusal rate", "overall", "refused"),
        ("Latency p50 (s)", "overall", "latency_p50_s"),
    ]

    records = []
    for label, group, key in rows:
        r = rag.get(group, {}).get(key)
        b = base.get(group, {}).get(key) if base else None
        dec = 1 if "latency" in key else 3
        r = round(r, dec) if isinstance(r, (int, float)) else None
        b = round(b, dec) if isinstance(b, (int, float)) else None
        d = round(r - b, dec) if r is not None and b is not None else None
        records.append({"Metric": label, "Group": group, "RAG": r, "Baseline": b, "Delta": d})

    df = pd.DataFrame(records)
    out = report_file.with_name(report_file.name.replace("_report.json", "_summary.xlsx"))
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="summary", index=False)
        ws = writer.sheets["summary"]
        # readable column widths + numeric formats
        widths = {"A": 45, "B": 22, "C": 10, "D": 10, "E": 10}
        for col, w in widths.items():
            ws.column_dimensions[col].width = w
        for row in ws.iter_rows(min_row=2, min_col=3, max_col=5):
            for c in row:
                if c.value is not None:
                    c.number_format = "0.000" if "Latency" not in (ws.cell(c.row, 1).value or "") else "0.0"

    print(f"\n{'='*90}\nSUMMARY — {report_file.name}\n{'='*90}\n")
    print(df.to_string(index=False))
    print(f"\nSaved summary sheet in {out}")


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

    if value in ("latest", "latest_baseline"):
        prefix = "run_collect_" if value == "latest" else "run_baseline_"
        files = [f for f in sorted(RESULTS_DIR.glob(f"{prefix}*.json"))
                 if not f.name.endswith("_report.json")]
        if not files:
            raise FileNotFoundError(f"No {prefix}*.json run files in {RESULTS_DIR} yet")
        return files[-1]
    p = Path(value)
    
    if not p.exists() and not p.is_absolute() and str(p.parent) == ".":
        candidate = RESULTS_DIR / value
        if candidate.exists():
            return candidate
    return p


def find_latest_baseline():
    """Most recent baseline run file, or None if no baseline has been collected."""

    files = [f for f in sorted(RESULTS_DIR.glob("run_baseline_*.json"))
             if not f.name.endswith("_report.json")]
    return files[-1] if files else None


def resolve_report_arg(value: str | None):
    if value in (None, "latest"):
        files = sorted(RESULTS_DIR.glob("run_collect_*_report.json"))
        if not files:
            raise FileNotFoundError(f"No run_collect_*_report.json reports in {RESULTS_DIR} yet")
        return files[-1]
    p = resolve_run_arg(value)
    if p.exists() and p.name.endswith("_report.json"):
        return p
    return p.with_name(p.stem + "_report.json")


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
    parser.add_argument("--mode", required=True, choices=["collect", "evaluate", "baseline", "report", "summary"])
    parser.add_argument("--domain", default="travel_insurance")
    parser.add_argument("--test-set", default=str(PROJECT_ROOT / "data" / "evaluation" / "qna_dataset.json"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--run-file", help="Run JSON or 'latest' / 'latest_baseline' (evaluate/report)")
    parser.add_argument("--baseline-file", help="Baseline run JSON or 'latest_baseline' (report)")
    parser.add_argument("--resume", default=None, help="Run file to continue (collect/baseline)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default=None, help="Comma-separated test ids")
    parser.add_argument("--cheap-only", action="store_true",
                    help="Skip RAGAS metrics (fast, no Ollama evaluator needed)")
    parser.add_argument("--top-k", type=int, default=None,
                    help="Override config retrieval.top_k for this run")
    args = parser.parse_args()

    config = load_config(args.config)
    test_set = load_test_set(Path(args.test_set))
    ids = args.ids.split(",") if args.ids else None
    resume = Path(args.resume) if args.resume else None

    if args.mode == "collect":
        if args.top_k is not None:
            config.setdefault("retrieval", {})["top_k"] = args.top_k
        collect(config, args.domain, test_set, args.limit, ids, resume, args.cheap_only)
    elif args.mode == "baseline":
        baseline(config, args.domain, test_set, args.limit, ids, resume, args.cheap_only)
    elif args.mode == "evaluate":
        if not args.run_file:
            parser.error("--run-file required (path or 'latest')")
        evaluate(config, test_set, resolve_run_arg(args.run_file), args.cheap_only)
    elif args.mode == "report":
        if not args.run_file:
            parser.error("--run-file required (path or 'latest')")
        report(
            resolve_run_arg(args.run_file),
            resolve_run_arg(args.baseline_file) if args.baseline_file else None,
        )
    elif args.mode == "summary":
        summary(resolve_report_arg(args.run_file))


if __name__ == "__main__":
    main()