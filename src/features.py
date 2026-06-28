"""Feature extraction, honeypot detection, and disqualifier penalties."""

from __future__ import annotations

import re
from datetime import date

AI_TITLE_TERMS = (
    "ai engineer", "ml engineer", "machine learning", "applied scientist",
    "applied ml", "nlp engineer", "research engineer", "ai/ml", "deep learning",
    "data scientist", "mlops", "search engineer", "relevance engineer",
)

ADJACENT_TITLE_TERMS = (
    "data engineer", "analytics engineer", "backend engineer", "software engineer",
    "platform engineer", "search", "recommendation",
)

IR_EVIDENCE = (
    "retrieval", "ranking", "rank ", "re-rank", "rerank", "recommendation",
    "recommender", "recsys", "search", "semantic search", "vector search",
    "embedding", "embeddings", "bm25", "hybrid retrieval", "nearest neighbor",
    "ann ", "learning to rank", "learning-to-rank", "ltr", "relevance",
    "personalization", "matching system",
)

VECTOR_INFRA = (
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "pgvector", "vector database", "vector db", "vector index",
)

EMBED_MODELS = (
    "sentence-transformers", "sentence transformers", "bge", " e5 ", "openai embedding",
    "embedding model", "encoder", "bi-encoder", "cross-encoder",
)

EVAL_TERMS = (
    "ndcg", "mrr", "map@", "precision@", "recall@", "a/b test", "ab test",
    "offline metric", "online metric", "eval framework", "evaluation framework",
    "offline-to-online", "recall@k", "hit rate",
)

LLM_TERMS = (
    "fine-tun", "lora", "qlora", "peft", "llm", "rag ", "retrieval augmented",
    "retrieval-augmented", "prompt", "instruction tun",
)

PROD_TERMS = (
    "production", "deployed", "real users", "at scale", "latency", "throughput",
    "served", "serving", "in production", "shipped",
)

CONSULTING_FIRMS = (
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "ltimindtree", "mindtree", "mphasis",
    "deloitte", "genpact",
)

RESEARCH_ENV = (
    "research lab", "phd", "postdoc", "post-doc", "university", "iit ", "iisc",
    "academic", "research scientist", "research-only", "publication",
)

FRAMEWORK_HYPE = ("langchain", "autogpt", "auto-gpt", "llamaindex", "llama-index")

PREFERRED_CITIES = (
    "pune", "noida", "delhi", "new delhi", "gurgaon", "gurugram", "ncr",
    "hyderabad", "mumbai", "bangalore", "bengaluru",
)

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

TODAY = date(2026, 6, 28)


def _parse_date(s):
    if not s or not isinstance(s, str):
        return None
    m = _DATE_RE.match(s)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _any(terms, text):
    return any(t in text for t in terms)


def _count(terms, text):
    return sum(1 for t in terms if t in text)


def _career_text(cand):
    parts = []
    for h in cand.get("career_history", []):
        parts.append(h.get("title", ""))
        parts.append(h.get("description", ""))
        parts.append(h.get("industry", ""))
    return " ".join(parts).lower()


def _full_text(cand):
    p = cand.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", ""), p.get("current_title", "")]
    parts.append(_career_text(cand))
    return " ".join(parts).lower()


def honeypot_flags(cand):
    flags = []
    p = cand.get("profile", {})
    yoe = p.get("years_of_experience", 0) or 0

    total_months = sum(h.get("duration_months", 0) or 0 for h in cand.get("career_history", []))
    if yoe > 0 and total_months > (yoe + 3) * 12 + 6:
        flags.append("tenure_exceeds_experience")

    for h in cand.get("career_history", []):
        if (h.get("duration_months", 0) or 0) > (yoe * 12) + 18 and yoe > 0:
            flags.append("role_longer_than_career")
            break

    impossible_skill = 0
    for s in cand.get("skills", []):
        prof = s.get("proficiency")
        dur = s.get("duration_months", None)
        if prof in ("expert", "advanced") and dur is not None and dur == 0:
            impossible_skill += 1
    if impossible_skill >= 3:
        flags.append("expert_skill_zero_duration")

    for e in cand.get("education", []):
        sy, ey = e.get("start_year"), e.get("end_year")
        if isinstance(sy, int) and isinstance(ey, int) and ey < sy:
            flags.append("education_dates_reversed")
            break

    for h in cand.get("career_history", []):
        sd = _parse_date(h.get("start_date"))
        ed = _parse_date(h.get("end_date"))
        if sd and ed and ed < sd:
            flags.append("role_end_before_start")
            break

    return flags


def is_honeypot(cand):
    return len(honeypot_flags(cand)) > 0


def title_fit(cand):
    title = cand.get("profile", {}).get("current_title", "").lower()
    headline = cand.get("profile", {}).get("headline", "").lower()
    ident = title + " " + headline

    if _any(AI_TITLE_TERMS, title):
        base = 1.0
        if "junior" in title:
            base = 0.55
        if any(s in title for s in ("senior", "staff", "lead", "principal")):
            base = min(1.0, base + 0.05)
        return base
    if _any(AI_TITLE_TERMS, ident):
        return 0.7
    if _any(ADJACENT_TITLE_TERMS, title):
        return 0.45
    return 0.05


def career_evidence(cand):
    text = _full_text(cand)

    ir = min(1.0, _count(IR_EVIDENCE, text) / 4.0)
    infra = min(1.0, _count(VECTOR_INFRA, text) / 2.0)
    embed = min(1.0, _count(EMBED_MODELS, text) / 2.0)
    evals = min(1.0, _count(EVAL_TERMS, text) / 2.0)
    prod = min(1.0, _count(PROD_TERMS, text) / 3.0)

    score = (
        0.34 * ir
        + 0.18 * infra
        + 0.12 * embed
        + 0.14 * evals
        + 0.22 * prod
    )
    return min(1.0, score), {
        "ir": ir, "infra": infra, "embed": embed, "eval": evals, "prod": prod,
    }


def product_company_ratio(cand):
    history = cand.get("career_history", [])
    if not history:
        return 0.5
    services_months = 0
    total_months = 0
    for h in history:
        m = h.get("duration_months", 0) or 0
        total_months += m
        comp = (h.get("company", "") + " " + h.get("industry", "")).lower()
        if _any(CONSULTING_FIRMS, comp) or "it services" in comp or "consulting" in comp:
            services_months += m
    if total_months == 0:
        return 0.5
    return 1.0 - (services_months / total_months)


def seniority_fit(cand):
    yoe = cand.get("profile", {}).get("years_of_experience", 0) or 0
    if 6 <= yoe <= 8:
        return 1.0
    if 5 <= yoe <= 9:
        return 0.9
    if 4 <= yoe < 5 or 9 < yoe <= 10:
        return 0.65
    if 3 <= yoe < 4 or 10 < yoe <= 12:
        return 0.4
    if yoe > 12:
        return 0.2
    return 0.15


def location_fit(cand):
    p = cand.get("profile", {})
    loc = p.get("location", "").lower()
    country = p.get("country", "").lower()
    relocate = cand.get("redrob_signals", {}).get("willing_to_relocate", False)

    if country != "india":
        return 0.5 if relocate else 0.25
    if any(c in loc for c in ("pune", "noida")):
        return 1.0
    if _any(PREFERRED_CITIES, loc):
        return 0.85
    return 0.7 if relocate else 0.55


def disqualifier_penalty(cand):
    penalty = 0.0
    reasons = []
    text = _full_text(cand)
    title = cand.get("profile", {}).get("current_title", "").lower()

    ai_skill_count = 0
    for s in cand.get("skills", []):
        nm = s.get("name", "").lower()
        if any(k in nm for k in (
            "rag", "pinecone", "weaviate", "qdrant", "milvus", "faiss", "llm",
            "lora", "qlora", "embedding", "vector search", "semantic search",
            "fine-tuning", "transformer", "pytorch", "tensorflow", "diffusion",
            "reinforcement learning", "recommendation", "nlp", "bert",
        )):
            ai_skill_count += 1
    has_ai_identity = _any(AI_TITLE_TERMS, title)
    ev_score, _ = career_evidence(cand)
    if ai_skill_count >= 5 and not has_ai_identity and ev_score < 0.18:
        penalty += 0.6
        reasons.append("keyword_stuffer")

    history = cand.get("career_history", [])
    if history:
        all_services = all(
            _any(CONSULTING_FIRMS, (h.get("company", "") + " " + h.get("industry", "")).lower())
            or "it services" in (h.get("industry", "").lower())
            for h in history
        )
        if all_services:
            penalty += 0.3
            reasons.append("consulting_only_career")

    if _any(RESEARCH_ENV, text) and not _any(PROD_TERMS, text):
        penalty += 0.25
        reasons.append("research_without_production")

    if _any(FRAMEWORK_HYPE, text) and ev_score < 0.2 and not has_ai_identity:
        penalty += 0.2
        reasons.append("framework_hype_only")

    short_stints = sum(
        1 for h in history
        if (h.get("duration_months", 0) or 0) < 20 and not h.get("is_current", False)
    )
    if len(history) >= 3 and short_stints >= 3:
        penalty += 0.15
        reasons.append("title_chaser_short_stints")

    return min(penalty, 0.9), reasons


def behavioral_multiplier(cand):
    sig = cand.get("redrob_signals", {})
    reasons = {}

    resp = sig.get("recruiter_response_rate", 0.0) or 0.0
    resp_factor = 0.6 + 0.5 * resp
    reasons["response_rate"] = resp

    last = _parse_date(sig.get("last_active_date"))
    if last:
        days = (TODAY - last).days
        if days <= 14:
            rec = 1.05
        elif days <= 45:
            rec = 1.0
        elif days <= 90:
            rec = 0.9
        elif days <= 180:
            rec = 0.75
        else:
            rec = 0.55
    else:
        rec = 0.85
    reasons["days_since_active"] = (TODAY - last).days if last else None

    otw = 1.05 if sig.get("open_to_work_flag") else 0.95

    icr = sig.get("interview_completion_rate", None)
    interview = 1.0
    if isinstance(icr, (int, float)):
        interview = 0.9 + 0.15 * max(0.0, min(1.0, icr))

    notice = sig.get("notice_period_days", None)
    notice_factor = 1.0
    if isinstance(notice, (int, float)):
        if notice <= 30:
            notice_factor = 1.03
        elif notice <= 60:
            notice_factor = 0.98
        elif notice <= 90:
            notice_factor = 0.93
        else:
            notice_factor = 0.88
    reasons["notice_period_days"] = notice

    mult = resp_factor * rec * otw * interview * notice_factor
    mult = max(0.45, min(1.15, mult))
    return mult, reasons
