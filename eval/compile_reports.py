"""Compile evaluation reports into a single Excel workbook, one sheet per report.

Usage:
    python eval/compile_reports.py                 # all reports in eval/results
    python eval/compile_reports.py --latest 5      # only the 5 most recent
    python eval/compile_reports.py -o my.xlsx      # custom output path

Sheets:
    overview          - one row per report (overall metrics), newest first
    <run name>        - per-group metrics for that report (RAG, and baseline
                        comparison columns when the report carries one)
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUT_PATH = RESULTS_DIR / "all_reports.xlsx"

METRIC_ORDER = [
    "n",
    "faithfulness",
    "answer_correctness",
    "context_precision",
    "ookb_correct",
    "retrieval_hit_at_k",
    "retrieval_precision_at_k",
    "attribution_precision",
    "refused",
    "latency_p50_s",
]


def sheet_name(run_name: str, used: set) -> str:
    """Build a unique Excel-safe sheet name (<=31 chars, no []:*?/\\)."""

    name = run_name.replace("run_collect_travel_insurance_", "collect_")
    name = name.replace("run_baseline_travel_insurance_", "baseline_")
    for ch in "[]:*?/\\":
        name = name.replace(ch, "_")
    name = name[:31]
    base, i = name, 2
    while name in used:
        suffix = f"~{i}"
        name = base[: 31 - len(suffix)] + suffix
        i += 1
    used.add(name)
    return name


def report_rows(data: dict) -> pd.DataFrame:
    """One row per group; baseline metrics get a 'baseline_' prefix column."""

    rows = []
    groups = list(dict.fromkeys(list(data.get("rag", {})) + list(data.get("baseline", {}))))
    for g in groups:
        row = {"group": g}
        rag = data.get("rag", {}).get(g, {})
        base = data.get("baseline", {}).get(g, {})
        for k in METRIC_ORDER:
            if k in rag:
                row[k] = rag[k]
        for k in METRIC_ORDER:
            if k in base:
                row[f"baseline_{k}"] = base[k]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.set_index("group").reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--latest", type=int, default=0, help="limit to N most recent reports")
    parser.add_argument("-o", "--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    files = sorted(args.results_dir.glob("*_report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise SystemExit(f"No *_report.json files found in {args.results_dir}")
    if args.latest:
        files = files[: args.latest]

    overview = []
    per_report = {}
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        run_name = data.get("run_file", f.name).replace(".json", "")
        df = report_rows(data)
        per_report[run_name] = df
        overall = data.get("rag", {}).get("overall") or data.get("baseline", {}).get("overall") or {}
        overview.append({"report": run_name, "source": "rag" if "rag" in data else "baseline", **overall})

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        used = set()
        # overview sheet first, newest on top
        ov = pd.DataFrame(overview)
        ov_cols = ["report", "source"] + [c for c in METRIC_ORDER if c in ov.columns]
        ov[ov_cols].to_excel(writer, sheet_name="overview", index=False)
        used.add("overview")

        for run_name, df in per_report.items():
            df.to_excel(writer, sheet_name=sheet_name(run_name, used), index=False)

        # widen columns a little for readability
        for ws in writer.book.worksheets:
            for col_cells in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
                letter = col_cells[0].column_letter
                ws.column_dimensions[letter].width = min(max(width + 2, 10), 42)

    print(f"Compiled {len(files)} report(s) into {args.out}")
    print("Sheets:", ", ".join(ws.title for ws in writer.book.worksheets))


if __name__ == "__main__":
    main()
