"""Run the RAG evaluation dataset and write machine-readable results.

Example (from the repository root):
    python -m src.evaluate
    python -m src.evaluate --insurer ALLIANZ --limit 3

The runner deliberately does not call ingest.py: ingestion deletes and rebuilds
the Chroma collection. It evaluates the existing data/vector_db collection.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.query import ask


ROOT = Path(__file__).resolve().parents[1]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def answer_overlap(answer: str, golden: str) -> float:
    """A transparent baseline score; use an LLM judge later for nuance."""
    expected = _tokens(golden)
    actual = _tokens(answer)
    return round(len(expected & actual) / len(expected), 3) if expected else 0.0


def evaluate_case(case: dict, db_path: str, collection: str, top_k: int) -> dict:
    result = ask(case["question"], db_path, collection, case["insurer"], top_k)
    pages = sorted({int(source["page"]) for source in result.sources})
    expected_pages = sorted(int(page) for page in case.get("relevant_pages", []))
    page_hit = bool(set(pages) & set(expected_pages)) if expected_pages else None
    return {
        **case,
        "answer": result.answer,
        "retrieved_pages": pages,
        "retrieved_sources": result.sources,
        "retrieval_page_hit": page_hit,
        "answer_token_overlap": answer_overlap(result.answer, case["golden_answer"]),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def summary(results: list[dict]) -> dict:
    def avg(values):
        return round(sum(values) / len(values), 3) if values else None

    factual = [r for r in results if r["question_type"] == "factual"]
    reasoning = [r for r in results if r["question_type"] == "reasoning"]
    ook = [r for r in results if r["question_type"] == "out_of_knowledge_base"]
    in_kb = [r for r in results if r["relevant_pages"]]
    hits = [r["retrieval_page_hit"] for r in in_kb if r["retrieval_page_hit"] is not None]
    return {
        "total": len(results),
        "by_type": {kind: len([r for r in results if r["question_type"] == kind])
                     for kind in ("factual", "reasoning", "out_of_knowledge_base")},
        "retrieval_page_hit_rate": avg([int(hit) for hit in hits]),
        "answer_token_overlap": avg([r["answer_token_overlap"] for r in results]),
        "by_type_answer_token_overlap": {
            kind: avg([r["answer_token_overlap"] for r in group])
            for kind, group in (("factual", factual), ("reasoning", reasoning), ("out_of_knowledge_base", ook))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "data/evaluation/qna_dataset.json"))
    parser.add_argument("--output", default=str(ROOT / "data/evaluation/eval_results.json"))
    parser.add_argument("--db-path", default=str(ROOT / "data/vector_db"))
    parser.add_argument("--collection", default="Travel_Insurance")
    parser.add_argument("--insurer", help="Only evaluate one insurer")
    parser.add_argument("--limit", type=int, help="Only run the first N selected cases")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Keep existing case results in output")
    args = parser.parse_args()

    cases = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if args.insurer:
        cases = [case for case in cases if case["insurer"].upper() == args.insurer.upper()]
    if args.limit:
        cases = cases[:args.limit]

    output = Path(args.output)
    existing = {}
    if args.resume and output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        existing = {item["id"]: item for item in old.get("results", [])}

    results = []
    for index, case in enumerate(cases, 1):
        if case["id"] in existing:
            results.append(existing[case["id"]])
            print(f"[{index}/{len(cases)}] {case['id']} (resumed)")
            continue
        print(f"[{index}/{len(cases)}] {case['id']} {case['question']}")
        results.append(evaluate_case(case, args.db_path, args.collection, args.top_k))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"summary": summary(results), "results": results}, indent=2), encoding="utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary(results), "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary(results), indent=2))


if __name__ == "__main__":
    main()
