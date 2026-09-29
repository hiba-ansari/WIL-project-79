"""
Compile the overnight sweep results into one Excel workbook.

Reads every run file produced by eval/run_sweep.py (i.e. files following the
naming convention: run_collect_<domain>_k_<k>_t_<temp>_<ts>.json and
run_baseline_<domain>_t_<temp>_<ts>.json) and writes:

    eval/results/sweep_summary.xlsx

Sheets:
    grid_overview  one row per (mode, k, temperature): means for every metric
    hit_at_k       retrieval_hit_at_k by k x temperature (pivot)
    refusals_ookb  refusal rate + out-of-scope correctness by config (pivot)
    rag_vs_baseline  best RAG config vs matching-temperature baseline
    per_item       every item x config: scores + answer excerpt (for spot checks)

Old pre-sweep run files (no _t_ in the name) are ignored.

Usage:
    python eval/aggregate_sweep.py
    python eval/aggregate_sweep.py -o eval/results/sweep_summary_20260930.xlsx
"""

import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

COLLECT_RE = re.compile(r"^run_collect_travel_insurance_k_(\d+)_t([\d.]+)_\d{8}_\d{6}\.json$")
BASELINE_RE = re.compile(r"^run_baseline_travel_insurance_t([\d.]+)_\d{8}_\d{6}\.json$")

METRIC_KEYS = ["faithfulness", "answer_correctness", "context_precision",
               "ookb_correct", "retrieval_hit_at_k", "retrieval_precision_at_k",
               "attribution_precision", "refused"]
CATS = ["factual", "reasoning", "out_of_knowledge_base"]


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def load_runs():
    """Latest run file per (mode, k, temperature) config."""
    runs = {}
    for f in sorted(RESULTS_DIR.glob("run_collect_*.json")):
        m = COLLECT_RE.match(f.name)
        if not m:
            continue
        runs[("collect", int(m.group(1)), float(m.group(2)))] = f  # sorted to get latest run
    for f in sorted(RESULTS_DIR.glob("run_baseline_*.json")):
        m = BASELINE_RE.match(f.name)
        if not m:
            continue
        runs[("baseline", None, float(m.group(1)))] = f
    return runs


def main(out_path: Path):
    runs = load_runs()
    if not runs:
        sys.exit("No sweep run files found in eval/results/ - did the sweep complete?")

    grid_rows, item_rows = [], []
    for (mode, k, t), path in sorted(runs.items(), key=lambda x: (x[0][0], x[0][1] or 0, x[0][2])):
        run = json.loads(path.read_text(encoding="utf-8"))
        entries = run["entries"]
        n = len(entries)
        row = {"mode": mode, "top_k": k, "temperature": t, "n_items": n,
               "run_file": path.name}
        groups = {"overall": entries,
                  **{c: [e for e in entries if e["question_type"] == c] for c in CATS}}
        for gname, ge in groups.items():
            for key in METRIC_KEYS:
                val = mean_or_none([e.get("scores", {}).get(key) for e in ge])
                row[f"{gname}_{key}"] = val
        lats = [e["latency_s"] for e in entries if "latency_s" in e]
        row["latency_p50_s"] = round(statistics.median(lats), 2) if lats else None
        grid_rows.append(row)

        for e in entries:
            item = {"run_file": path.name, "mode": mode, "top_k": k, "temperature": t,
                    "id": e["id"], "question_type": e["question_type"],
                    "answer_excerpt": (e["answer"] or "")[:200]}
            item.update({key: e.get("scores", {}).get(key) for key in METRIC_KEYS})
            item_rows.append(item)

    grid_df = pd.DataFrame(grid_rows)

    collect = grid_df[grid_df["mode"] == "collect"]
    hit_df = collect.pivot_table(index="top_k", columns="temperature",
                                 values="overall_retrieval_hit_at_k")
    prec_df = collect.pivot_table(index="top_k", columns="temperature",
                                  values="overall_retrieval_precision_at_k")
    ref_df = grid_df.pivot_table(index=["mode", "top_k"], columns="temperature",
                                 values="overall_refused")
    ookb_df = grid_df.pivot_table(index=["mode", "top_k"], columns="temperature",
                                  values="out_of_knowledge_base_ookb_correct")

    # RAG vs baseline: each collect config paired with baseline at same temperature
    cmp_rows = []
    base_by_t = {row["temperature"]: row
                 for row in grid_df[grid_df["mode"] == "baseline"].to_dict("records")}
    for r in collect.to_dict("records"):
        b = base_by_t.get(r["temperature"])
        row = {"top_k": r["top_k"], "temperature": r["temperature"]}
        for key, label in [("overall_answer_correctness", "answer_correctness"),
                           ("overall_faithfulness", "faithfulness"),
                           ("out_of_knowledge_base_ookb_correct", "ookb_correct"),
                           ("overall_refused", "refused")]:
            row[f"RAG_{label}"] = r.get(key)
            bv = b.get(key) if b is not None else None
            row[f"base_{label}"] = bv
            rv = r.get(key)
            row[f"delta_{label}"] = (round(rv - bv, 4)
                                     if isinstance(rv, (int, float)) and isinstance(bv, (int, float))
                                     else None)
        cmp_rows.append(row)
    cmp_df = pd.DataFrame(cmp_rows)

    # machine-readable version of the workbook allowing for matplotlib graphs to be made too
    def records(df):
        d = df.reset_index() # reset_index() keeps pivot tables JSON-safe (since tuple index levels become plain values in records)
        d.columns = [str(c) for c in d.columns]
        return d.to_dict(orient="records")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "grid_overview": grid_rows,
        "hit_at_k": records(hit_df),
        "precision_at_k": records(prec_df),
        "refused": records(ref_df),
        "ookb_correct": records(ookb_df),
        "rag_vs_baseline": records(cmp_df),
        "per_item": item_rows,
    }, indent=2, default=str), encoding="utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        grid_df.to_excel(writer, sheet_name="grid_overview", index=False)
        hit_df.to_excel(writer, sheet_name="hit_at_k")
        prec_df.to_excel(writer, sheet_name="hit_at_k", startcol=hit_df.shape[1] + 3)
        ref_df.to_excel(writer, sheet_name="refusals_ookb")
        ookb_df.to_excel(writer, sheet_name="refusals_ookb", startcol=ref_df.shape[1] + 3)
        cmp_df.to_excel(writer, sheet_name="rag_vs_baseline", index=False)
        pd.DataFrame(item_rows).to_excel(writer, sheet_name="per_item", index=False)
        # readable widths
        for name, ws in writer.sheets.items():
            for col, letter in zip("ABCDEFGHIJKLMNOPQRSTU", "ABCDEFGHIJKLMNOPQRSTU"):
                ws.column_dimensions[letter].width = 14
            ws.column_dimensions["A"].width = 22

    print(f"Saved {out_path}")
    print(f"Saved {json_path}")
    print(f"\nconfigs found: {len(runs)} "
          f"({len(collect)} collect, {len(runs) - len(collect)} baseline)")
    print("\ngrid_overview:")
    show = ["mode", "top_k", "temperature", "n_items",
            "overall_answer_correctness", "overall_faithfulness",
            "overall_retrieval_hit_at_k", "out_of_knowledge_base_ookb_correct",
            "overall_refused"]
    print(grid_df[[c for c in show if c in grid_df.columns]].to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output", default=str(RESULTS_DIR / "sweep_summary.xlsx"))
    args = parser.parse_args()
    main(Path(args.output))
