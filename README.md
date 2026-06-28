# Redrob — Intelligent Candidate Discovery & Ranking

A CPU-only, network-free, **explainable** ranker that selects the top 100 candidates
from a 100,000-profile pool for the *Senior AI Engineer — Founding Team* job description
in the Redrob hackathon.

## TL;DR

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runs end-to-end in **~80 seconds for 100K candidates** on a CPU laptop, no GPU,
no network, no model downloads — comfortably inside the 5-minute / 16 GB budget.
Output is a spec-valid CSV (`candidate_id,rank,score,reasoning`).

## The core idea

The hackathon authors are explicit (in `job_description.md`) that the *wrong* answer
is "find candidates whose skills section contains the most AI keywords" — that is a
**trap deliberately built into the dataset**. A Marketing Manager who lists `RAG`,
`Pinecone`, and `LoRA` as skills is not a fit; a candidate without those exact words
who *built a recommendation system at a product company* is.

So this ranker reads **structured evidence**, not keywords:

| What we read | Why |
|---|---|
| **Role identity** (title + headline) | Primary anti-keyword-stuffer signal. A "Marketing Manager" scores ~0 on identity no matter what skills they list. |
| **Career evidence** (role descriptions + summary) | Detects genuine *retrieval / ranking / recommendation / search* work, vector infra, eval literacy (NDCG/MRR/A-B), and production/scale language — exactly the JD's "things you absolutely need". |
| **Product vs services** ratio | JD strongly prefers product-company applied ML over consulting/services. |
| **Seniority** | Soft fit to the JD's 5–9yr band (ideal 6–8). |
| **Location** | Pune/Noida preferred; Indian Tier-1 welcome; outside-India down-weighted (no visa sponsorship). |
| **IR-depth bonus** | Separates a true search/ranking engineer from a generic data scientist. |

On top of relevance we apply the JD's **explicit disqualifiers** as penalties
(keyword-stuffing in a non-AI role, consulting-only careers, research-without-production,
LangChain-only hype, title-churn), then a **behavioral availability multiplier**
(recruiter response rate, recency, open-to-work, notice period, interview reliability),
because "a perfect-on-paper candidate who hasn't logged in for 6 months ... is not
actually available."

**Honeypots** (`submission_spec.md` §7 — ~80 profiles with impossible internals) are
detected via internal contradictions (tenure exceeding stated experience, expert
proficiency on skills with 0 months of use, reversed dates) and forced to score 0.
Our top-100 contains **0 honeypots** (Stage-3 filter requires < 10%).

### Scoring formula

```
relevance  = Σ wᵢ · featureᵢ          # six JD-aligned features, weights sum to 1
relevance -= disqualifier_penalty      # subtractive, JD's explicit "do NOT want"
final      = max(0, relevance) · behavioral_multiplier
final      = 0                         # if honeypot
```

Relevance (fit) is additive; availability is multiplicative — mirroring how the JD
frames them. The whole thing is a transparent function of named features, which is
what lets every score be turned into grounded, per-candidate **reasoning** for the
Stage-4 manual review (no hallucinated skills — reasoning is built only from facts
present in the candidate record).

## Project layout

```
rank.py                 # CLI entrypoint: stream -> score -> top-100 CSV
app.py                  # Streamlit sandbox demo (HF Spaces / Streamlit Cloud)
src/
  features.py           # feature extraction, honeypot & disqualifier detection
  scoring.py            # weighted scoring + grounded reasoning generation
tests/
  test_ranker.py        # encodes the JD's rules as assertions
requirements.txt        # stdlib only for ranking (pytest optional for tests)
submission_metadata.yaml
```

## Reproduce

```bash
# 1. (if gzipped) unpack the pool
gunzip -k candidates.jsonl.gz

# 2. produce the submission
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 3. validate against the official validator
python validate_submission.py ./submission.csv      # -> "Submission is valid."

# 4. (optional) run the test suite
python tests/test_ranker.py                          # or: python -m pytest tests/
```

`rank.py` also accepts a `.gz` path directly (`--candidates ./candidates.jsonl.gz`).

## Compute constraints (all satisfied)

| Constraint | Limit | This solution |
|---|---|---|
| Runtime | ≤ 5 min | ~80 s for 100K |
| Memory | ≤ 16 GB | streams line-by-line; well under |
| Compute | CPU only | pure Python, no GPU |
| Network | off | no API calls, no model downloads |
| Disk | ≤ 5 GB | none beyond the output CSV |

## Sandbox

`app.py` is a Streamlit app for the mandatory sandbox link. It accepts a small
JSONL/JSON sample (e.g. `sample_candidates.json`), runs the ranker end-to-end on CPU,
and offers the ranked CSV for download. Deploy to HuggingFace Spaces or Streamlit
Cloud (free tier), or run `streamlit run app.py` locally.

## Notes on AI tool usage

Built with Claude (Claude Code) for design, feature work, and review. The ranking
code itself calls **no** LLM and sends **no** candidate data to any hosted service at
ranking time — every scoring decision is explicit, inspectable Python. See
`submission_metadata.yaml`.
