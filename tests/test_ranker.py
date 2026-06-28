"""Sanity tests for the ranker's core judgments.

Run with:  python -m pytest tests/  (or python tests/test_ranker.py)

These encode the JD's explicit rules as assertions so a future refactor can't
silently regress them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scoring import score_candidate  # noqa: E402
from src.features import is_honeypot, title_fit  # noqa: E402


def _base(**over):
    cand = {
        "candidate_id": "CAND_0000000",
        "profile": {
            "anonymized_name": "Test",
            "headline": "h",
            "summary": "s",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "ML Engineer",
            "current_company": "Acme",
            "current_company_size": "201-500",
            "current_industry": "Software",
        },
        "career_history": [{
            "company": "Acme", "title": "ML Engineer", "start_date": "2020-01-01",
            "end_date": None, "duration_months": 40, "is_current": True,
            "industry": "Software", "company_size": "201-500",
            "description": "Built production retrieval and ranking systems with embeddings, "
                           "deployed to real users; evaluated with NDCG and A/B tests.",
        }],
        "education": [],
        "skills": [],
        "redrob_signals": {
            "recruiter_response_rate": 0.8, "last_active_date": "2026-06-10",
            "open_to_work_flag": True, "notice_period_days": 30,
            "interview_completion_rate": 0.9, "willing_to_relocate": True,
            "signup_date": "2019-01-01", "github_activity_score": 50,
        },
    }
    cand["profile"].update(over.get("profile", {}))
    for k, v in over.items():
        if k != "profile":
            cand[k] = v
    return cand


def test_strong_ai_engineer_scores_high():
    res = score_candidate(_base())
    assert res["score"] > 0.7, res


def test_keyword_stuffer_marketing_manager_is_penalised():
    # Marketing Manager with all the AI skills but no AI career evidence.
    stuffer = _base(
        profile={"current_title": "Marketing Manager", "summary": "Marketing leader."},
        career_history=[{
            "company": "BrandCo", "title": "Marketing Manager", "start_date": "2018-01-01",
            "end_date": None, "duration_months": 80, "is_current": True,
            "industry": "Marketing", "company_size": "51-200",
            "description": "Ran campaigns and managed social media.",
        }],
        skills=[{"name": n, "proficiency": "expert", "endorsements": 5}
                for n in ("RAG", "Pinecone", "LoRA", "Embeddings", "Vector Search", "NLP")],
    )
    strong = score_candidate(_base())
    weak = score_candidate(stuffer)
    assert weak["score"] < strong["score"] * 0.5, (weak["score"], strong["score"])


def test_honeypot_zeroed():
    # Expert proficiency, zero duration, on many skills -> impossible profile.
    hp = _base(skills=[
        {"name": n, "proficiency": "expert", "endorsements": 1, "duration_months": 0}
        for n in ("a", "b", "c", "d")
    ])
    assert is_honeypot(hp)
    assert score_candidate(hp)["score"] == 0.0


def test_low_availability_downweighted_not_zeroed():
    avail = score_candidate(_base())
    unavail = score_candidate(_base(redrob_signals={
        **_base()["redrob_signals"],
        "recruiter_response_rate": 0.03,
        "last_active_date": "2025-09-01",
        "open_to_work_flag": False,
    }))
    assert unavail["score"] < avail["score"]
    assert unavail["score"] > 0.0  # availability dampens, doesn't disqualify


def test_title_fit_marketing_near_zero():
    assert title_fit({"profile": {"current_title": "Marketing Manager", "headline": ""}}) < 0.1
    assert title_fit({"profile": {"current_title": "ML Engineer", "headline": ""}}) >= 0.9


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
