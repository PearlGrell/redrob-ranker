"""Scoring and reasoning.

    relevance = weighted sum of features - disqualifier penalty
    final     = max(0, relevance) * behavioral_multiplier   (0 if honeypot)
"""

from __future__ import annotations

from . import features as F

WEIGHTS = {
    "title_fit": 0.26,
    "career_evidence": 0.30,
    "product_company": 0.12,
    "seniority": 0.14,
    "location": 0.10,
    "ir_depth_bonus": 0.08,
}


def score_candidate(cand):
    hflags = F.honeypot_flags(cand)
    if hflags:
        return {
            "score": 0.0,
            "relevance": 0.0,
            "multiplier": 0.0,
            "honeypot": True,
            "honeypot_flags": hflags,
            "components": {},
            "evidence": {},
            "penalties": [],
            "behavior": {},
        }

    title = F.title_fit(cand)
    evidence, ev_parts = F.career_evidence(cand)
    product = F.product_company_ratio(cand)
    seniority = F.seniority_fit(cand)
    location = F.location_fit(cand)

    ir_depth = min(1.0, 0.5 * ev_parts["ir"] + 0.3 * ev_parts["infra"] + 0.2 * ev_parts["eval"])

    components = {
        "title_fit": title,
        "career_evidence": evidence,
        "product_company": product,
        "seniority": seniority,
        "location": location,
        "ir_depth_bonus": ir_depth,
    }

    relevance = sum(WEIGHTS[k] * v for k, v in components.items())

    penalty, pen_reasons = F.disqualifier_penalty(cand)
    relevance = max(0.0, relevance - penalty)

    mult, beh = F.behavioral_multiplier(cand)
    final = relevance * mult

    return {
        "score": final,
        "relevance": relevance,
        "multiplier": mult,
        "honeypot": False,
        "honeypot_flags": [],
        "components": components,
        "evidence": ev_parts,
        "penalties": pen_reasons,
        "behavior": beh,
    }


_SKILL_HINTS = (
    "retrieval", "ranking", "recommendation", "recsys", "search", "semantic",
    "vector", "embedding", "bm25", "rag", "nlp", "fine-tun", "lora", "qlora",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "ndcg",
    "mrr", "learning to rank", "personalization", "relevance",
)


def _relevant_snippets(cand, limit=3):
    found = []
    p = cand.get("profile", {})
    blob = (p.get("summary", "") + " ").lower()
    for h in cand.get("career_history", []):
        blob += (h.get("description", "") + " ").lower()
    for hint in _SKILL_HINTS:
        if hint in blob and hint not in found:
            found.append(hint)
        if len(found) >= limit:
            break
    return found


def build_reasoning(cand, result):
    p = cand.get("profile", {})
    title = p.get("current_title", "candidate")
    yoe = p.get("years_of_experience", 0)
    loc = p.get("location", "")
    country = p.get("country", "")
    sig = cand.get("redrob_signals", {})

    if result["honeypot"]:
        return (
            f"{title}, but profile has internal inconsistencies "
            f"({', '.join(result['honeypot_flags'])}); treated as a honeypot and excluded from fit."
        )

    comp = result["components"]
    parts = []

    yoe_str = f"{yoe:.1f}" if isinstance(yoe, (int, float)) else str(yoe)
    lead = f"{title} with {yoe_str} yrs"
    if loc:
        lead += f", based in {loc.split(',')[0]}{' (India)' if country == 'India' else f', {country}'}"
    parts.append(lead)

    snips = _relevant_snippets(cand)
    if comp["career_evidence"] >= 0.35 and snips:
        parts.append("career shows " + "/".join(snips[:3]) + " work")
    elif comp["career_evidence"] >= 0.2 and snips:
        parts.append("some " + "/".join(snips[:2]) + " exposure")
    elif comp["title_fit"] >= 0.7:
        parts.append("AI/ML role but thin retrieval-specific evidence in history")
    else:
        parts.append("limited evidence of search/ranking work the JD requires")

    resp = sig.get("recruiter_response_rate")
    days = result["behavior"].get("days_since_active")
    beh_bits = []
    if isinstance(resp, (int, float)):
        beh_bits.append(f"recruiter response {resp:.2f}")
    if isinstance(days, int):
        beh_bits.append(f"active {days}d ago")
    notice = sig.get("notice_period_days")
    if isinstance(notice, (int, float)):
        beh_bits.append(f"{int(notice)}d notice")
    if beh_bits:
        parts.append("; ".join(beh_bits))

    concerns = []
    if result["penalties"]:
        nice = {
            "keyword_stuffer": "AI skills listed but role/career don't back them up",
            "consulting_only_career": "entirely services/consulting background",
            "research_without_production": "research-leaning with little production signal",
            "framework_hype_only": "mostly framework/LangChain-level exposure",
            "title_chaser_short_stints": "several short stints (title-churn concern)",
        }
        concerns += [nice.get(x, x) for x in result["penalties"]]
    if comp["location"] < 0.5:
        concerns.append("outside India / relocation uncertain")
    if comp["seniority"] < 0.5:
        concerns.append("experience outside the 5-9yr band")
    if isinstance(resp, (int, float)) and resp < 0.2:
        concerns.append("low recruiter response rate (availability risk)")
    if concerns:
        parts.append("concerns: " + "; ".join(concerns[:2]))

    text = ". ".join(parts) + "."
    return text[:300]
