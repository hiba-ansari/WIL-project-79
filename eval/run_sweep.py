"""
Overnight sweep driver for the RAG evaluation.

Two grids:

  --sanity    cheap-only collect at k=<config default> for
              temperature in {0.1, 0.07, 0.02} to confirm the low end is
              behaviourally identical as expected.

  (default)   overnight full-RAGAS sweep:
              collect  k in {3, 5, 7} x temperature in {0.1, 0.5, 0.8} (9 runs)
              baseline temperature in {0.1, 0.5, 0.8}                  (3 runs)

Usage (from repo root, venv active, Ollama running):
    python eval/run_sweep.py --sanity
    python eval/run_sweep.py

Generate report afterwards:
    python eval/aggregate_sweep.py
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "eval" / "evaluation.py"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
DATASET = PROJECT_ROOT / "data" / "evaluation" / "qna_dataset.json"

DOMAIN = "travel_insurance"

SANITY_TEMPERATURES = [0.1, 0.07, 0.02]
SWEEP_TOP_K = [3, 5, 7]
SWEEP_TEMPERATURES = [0.1, 0.5, 0.8]

# helper functions

def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "sweep.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def n_items() -> int:
    return len(json.loads(DATASET.read_text(encoding="utf-8")))


def find_run_file(prefix: str):
    """Newest completed-or-partial run file matching prefix (no report files)."""
    files = [f for f in sorted(RESULTS_DIR.glob(f"{prefix}*.json"))
             if not f.name.endswith("_report.json")]
    return files[-1] if files else None


def entry_count(path: Path) -> int:
    try:
        return len(json.loads(path.read_text(encoding="utf-8"))["entries"])
    except Exception:
        return 0


def run_evaluation(args: list[str]) -> int:
    """One evaluation.py invocation as a subprocess; output goes to console + sweep.log."""
    cmd = [sys.executable, str(EVAL_SCRIPT)] + args
    log(f"    $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return proc.returncode


# grids

def sweep_config(mode: str, k: int | None, t: float, cheap_only: bool, total: int):
    """Run one (mode, k, temperature) config. Returns (status, run_file).

    status: 'skipped' | 'ok' | 'failed'
    """
    prefix = (f"run_collect_{DOMAIN}_k_{k}_t{t}_" if mode == "collect"
              else f"run_baseline_{DOMAIN}_t{t}_")
    existing = find_run_file(prefix)
    if existing and entry_count(existing) >= total:
        log(f"  SKIP {prefix}*-  already complete ({total}/{total}) in {existing.name}")
        return "skipped", existing
    if existing:
        log(f"  RESUME {existing.name} ({entry_count(existing)}/{total} done)")

    for attempt in (1, 2):
        args = ["--mode", mode]
        if mode == "collect":
            args += ["--top-k", str(k)]
        args += ["--temperature", str(t)]
        if cheap_only:
            args.append("--cheap-only")
        if existing:
            args += ["--resume", str(existing)]

        t0 = time.perf_counter()
        rc = run_evaluation(args)
        mins = (time.perf_counter() - t0) / 60

        done = find_run_file(prefix)
        if done and entry_count(done) >= total:
            log(f"  OK   {done.name}  ({mins:.1f} min)")
            return "ok", done
        log(f"  !!   {mode} k={k} t={t} attempt {attempt} incomplete "
            f"(rc={rc}, {mins:.1f} min)")
        existing = done  # resume whatever partial file exists on retry

    return "failed", None


def run_grid(configs: list[tuple], total: int):
    results = []
    t_start = time.perf_counter()
    for i, (mode, k, t, cheap) in enumerate(configs, 1):
        label = f"[{i}/{len(configs)}] {mode} k={k} t={t} {'cheap' if cheap else 'full-RAGAS'}"
        log(label)
        status, path = sweep_config(mode, k, t, cheap, total)
        results.append((mode, k, t, status, path))
    elapsed = (time.perf_counter() - t_start) / 60
    failed = [r for r in results if r[3] == "failed"]
    log(f"Sweep finished in {elapsed:.1f} min | "
        f"ok={sum(r[3] == 'ok' for r in results)} "
        f"skipped={sum(r[3] == 'skipped' for r in results)} "
        f"failed={len(failed)}")
    for mode, k, t, _, _ in failed:
        log(f"  FAILED config: mode={mode} k={k} t={t} - re-run this script to retry it")
    return 1 if failed else 0


# sanity report

def sanity_summary(total: int):
    """Compare answers across the sanity temperatures: how many items gave
    byte-identical answers, and per-config cheap metric means."""
    import statistics

    runs = {}
    for t in SANITY_TEMPERATURES:
        cand = [p for p in sorted(RESULTS_DIR.glob(f"run_collect_{DOMAIN}_k_*_t{t}_*.json"))
                if not p.name.endswith("_report.json")]
        if cand:
            runs[t] = json.loads(cand[-1].read_text(encoding="utf-8"))
    if len(runs) < 2:
        log("sanity: not enough runs to compare")
        return

    temps = sorted(runs)
    base = {e["id"]: e["answer"] for e in runs[temps[0]]["entries"]}
    log(f"sanity: answer stability vs t={temps[0]} ({total} items)")
    for t in temps[1:]:
        same = sum(1 for e in runs[t]["entries"] if base.get(e["id"]) == e["answer"])
        log(f"  t={t}: {same}/{total} answers identical "
            f"({(total - same) / total:.0%} changed)")

    for t in temps:
        es = runs[t]["entries"]
        m = lambda k: statistics.fmean([e["scores"][k] for e in es if e["scores"].get(k) is not None])
        log(f"  t={t}: hit_at_k={m('retrieval_hit_at_k'):.3f} "
            f"ookb_correct={m('ookb_correct'):.3f} refused={m('refused'):.3f}")


# main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sanity", action="store_true",
                        help=f"cheap-only stability check at temperatures {SANITY_TEMPERATURES}")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    if not DATASET.exists():
        sys.exit(f"dataset not found: {DATASET}")
    total = n_items()

    if args.sanity:
        configs = [("collect", None, t, True) for t in SANITY_TEMPERATURES]
    else:
        configs = [("collect", k, t, False) for k in SWEEP_TOP_K for t in SWEEP_TEMPERATURES]
        configs += [("baseline", None, t, False) for t in SWEEP_TEMPERATURES]

    # fill in k for sanity runs from config default
    import yaml
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    default_k = cfg.get("retrieval", {}).get("top_k", 5)
    configs = [(m, default_k if k is None else k, t, c) for m, k, t, c in configs]

    log(f"=== sweep start | mode={'sanity' if args.sanity else 'overnight'} | "
        f"{len(configs)} configs | {total} items/config ===")

    if args.dry_run:
        for m, k, t, c in configs:
            print(f"  {m:<9} k={k if m == 'collect' else '-':<3} t={t:<5} "
                  f"{'cheap-only' if c else 'full RAGAS'}")
        return 0

    rc = run_grid(configs, total)
    if args.sanity:
        sanity_summary(total)
        log("sanity done. If all three look identical, keep t=0.1 for production.")
    else:
        log("next: python eval/aggregate_sweep.py  ->  eval/results/sweep_summary.xlsx")
    return rc


if __name__ == "__main__":
    sys.exit(main())
