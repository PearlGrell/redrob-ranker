"""Sandbox demo app (HuggingFace Gradio Space).

Run locally:  python app.py
"""

import csv
import io
import json

import gradio as gr

from src.scoring import score_candidate, build_reasoning


def _load_candidates(file_obj, pasted_text):
    cands = []
    raw = ""
    if file_obj is not None:
        with open(file_obj, "r", encoding="utf-8") as f:
            raw = f.read()
    elif pasted_text and pasted_text.strip():
        raw = pasted_text

    raw = raw.strip()
    if not raw:
        return cands

    # Try a JSON array first (sample_candidates.json format).
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL (one object per line).
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cands.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return cands


def rank_sample(file_obj, pasted_text, top_n):
    cands = _load_candidates(file_obj, pasted_text)
    if not cands:
        return [], None, "No valid candidates found. Upload a JSONL/JSON sample or paste one."

    scored = []
    for c in cands:
        res = score_candidate(c)
        scored.append((round(res["score"], 4), c.get("candidate_id", "?"), c, res))
    scored.sort(key=lambda x: (-x[0], x[1]))

    table = []
    csv_rows = []
    for i, (score, cid, c, res) in enumerate(scored[: int(top_n)], start=1):
        reasoning = build_reasoning(c, res)
        title = c.get("profile", {}).get("current_title", "")
        table.append([i, cid, score, title, reasoning])
        csv_rows.append({"candidate_id": cid, "rank": i, "score": score, "reasoning": reasoning})

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    w.writeheader()
    for r in csv_rows:
        w.writerow(r)
    out_path = "ranked_sample.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())

    return table, out_path, f"Ranked {len(cands)} candidates; showing top {len(table)}."


with gr.Blocks(title="Redrob Candidate Ranker") as demo:
    gr.Markdown(
        "# Redrob — Intelligent Candidate Ranker\n"
        "Ranks candidates for the *Senior AI Engineer — Founding Team* JD. "
        "CPU-only, no network, explainable. Upload a JSONL/JSON sample "
        "(e.g. `sample_candidates.json`) or paste candidates below."
    )
    with gr.Row():
        file_in = gr.File(label="candidates.jsonl / .json (sample)", file_types=[".jsonl", ".json", ".txt"], type="filepath")
        text_in = gr.Textbox(label="…or paste JSONL / JSON here", lines=6)
    top_n = gr.Slider(5, 100, value=20, step=1, label="Top N")
    run = gr.Button("Rank", variant="primary")
    status = gr.Markdown()
    table = gr.Dataframe(
        headers=["rank", "candidate_id", "score", "title", "reasoning"],
        label="Ranking",
        wrap=True,
    )
    download = gr.File(label="Download ranked CSV")

    run.click(rank_sample, inputs=[file_in, text_in, top_n], outputs=[table, download, status])


if __name__ == "__main__":
    demo.launch()
