#!/usr/bin/env python3
"""Ranker entrypoint.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Reads the candidate pool (JSONL or .gz), scores every candidate, and writes the
top-100 submission CSV.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
import time

from src.scoring import score_candidate, build_reasoning

TOP_N = 100


def open_candidates(path):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_candidates(path):
    with open_candidates(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def rank(path, top_n=TOP_N, verbose=True):
    scored = []  # (score, secondary, candidate_id, cand, result)
    n = 0
    honeypots = 0
    t0 = time.time()

    for cand in iter_candidates(path):
        n += 1
        cid = cand.get("candidate_id")
        if not cid:
            continue
        res = score_candidate(cand)
        if res["honeypot"]:
            honeypots += 1
        secondary = res["relevance"]
        scored.append((res["score"], secondary, cid, cand, res))

    # Rank by rounded score (4dp), tie-break by candidate_id ascending (spec §3).
    def sort_key(x):
        return (-round(float(x[0]), 4), x[2])

    scored.sort(key=sort_key)

    leaders = scored[:top_n]

    if verbose:
        dt = time.time() - t0
        print(
            f"[rank] scored {n} candidates in {dt:.1f}s "
            f"({honeypots} honeypots zeroed); taking top {len(leaders)}",
            file=sys.stderr,
        )

    rows = []
    for i, (score, _sec, cid, cand, res) in enumerate(leaders, start=1):
        rows.append({
            "candidate_id": cid,
            "rank": i,
            "score": round(float(score), 4),
            "reasoning": build_reasoning(cand, res),
        })
    return rows


def enforce_monotonic(rows):
    """Clamp any rounding artifact so scores are non-increasing by rank (spec §3)."""
    prev = None
    for r in rows:
        if prev is not None and r["score"] > prev:
            r["score"] = prev
        prev = r["score"]
    return rows


def write_csv(rows, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Redrob candidate ranker")
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl(.gz)")
    ap.add_argument("--out", default="submission.csv", help="Output CSV path")
    ap.add_argument("--top", type=int, default=TOP_N, help="How many to rank (default 100)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    rows = rank(args.candidates, top_n=args.top, verbose=not args.quiet)
    rows = enforce_monotonic(rows)
    write_csv(rows, args.out)
    if not args.quiet:
        print(f"[rank] wrote {len(rows)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
