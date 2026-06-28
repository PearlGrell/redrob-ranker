"""Sandbox demo app (HuggingFace Spaces / Streamlit Cloud).

Run locally:  streamlit run app.py
"""

import io
import json

import streamlit as st

from src.scoring import score_candidate, build_reasoning

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob — Intelligent Candidate Ranker")
st.caption(
    "Ranks candidates for the *Senior AI Engineer — Founding Team* JD. "
    "CPU-only, no network, explainable. Upload a JSONL sample (one candidate per line)."
)

uploaded = st.file_uploader("candidates.jsonl (sample, <= 100 lines)", type=["jsonl", "json", "txt"])
top_n = st.slider("Top N", 5, 100, 20)

if uploaded is not None:
    text = io.TextIOWrapper(uploaded, encoding="utf-8")
    cands = []
    for line in text:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Allow a pretty-printed JSON array too (sample_candidates.json format).
        if isinstance(obj, list):
            cands.extend(obj)
        else:
            cands.append(obj)

    if not cands and uploaded:
        uploaded.seek(0)
        try:
            arr = json.load(io.TextIOWrapper(uploaded, encoding="utf-8"))
            if isinstance(arr, list):
                cands = arr
        except Exception:
            pass

    st.write(f"Loaded **{len(cands)}** candidates.")

    scored = []
    for c in cands:
        res = score_candidate(c)
        scored.append((round(res["score"], 4), c.get("candidate_id", "?"), c, res))
    scored.sort(key=lambda x: (-x[0], x[1]))

    rows = []
    for i, (score, cid, c, res) in enumerate(scored[:top_n], start=1):
        rows.append({
            "rank": i,
            "candidate_id": cid,
            "score": score,
            "title": c.get("profile", {}).get("current_title", ""),
            "reasoning": build_reasoning(c, res),
        })

    st.dataframe(rows, use_container_width=True)

    # Downloadable submission-format CSV.
    import csv
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    w.writeheader()
    for r in rows:
        w.writerow({
            "candidate_id": r["candidate_id"], "rank": r["rank"],
            "score": r["score"], "reasoning": r["reasoning"],
        })
    st.download_button("Download ranked CSV", buf.getvalue(),
                       file_name="ranked_sample.csv", mime="text/csv")
else:
    st.info("Upload a JSONL sample to see the ranking. "
            "You can use sample_candidates.json from the hackathon bundle.")
