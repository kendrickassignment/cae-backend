"""Evidence validation and deterministic regional scoring, independent of the LLM."""
import re
import unicodedata

from geography import COUNTRIES, CATALOG_VERSION

SCORING_VERSION = "asia-evidence-v1"
STATUS_POINTS = {"silent": 35, "excluded": 30, "deferred": 30, "partial": 15, "compliant": 0}
PATTERN_POINTS = {
    "silent_delisting": 20, "franchise_firewall": 15, "commitment_downgrade": 15,
    "geographic_exclusion": 12, "availability_clause": 10, "corporate_ghosting": 10,
    "timeline_deferral": 8, "hedging_language": 10,
}


def risk_level(score):
    if score is None:
        return "unknown"
    return "low" if score <= 30 else "medium" if score <= 55 else "high" if score <= 79 else "critical"


def _text(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def verify_evidence(raw, sources):
    """Check a quote against the exact source document and its local page, not all text."""
    verified = []
    seen = set()
    if not isinstance(raw, list):
        return verified
    for item in raw[:200]:
        if not isinstance(item, dict):
            continue
        source = item.get("source_document")
        page = item.get("page_number")
        quote = item.get("exact_quote")
        if not isinstance(source, str) or type(page) is not int or not isinstance(quote, str):
            continue
        normalized = _text(quote)
        if len(normalized) < 8 or normalized not in _text(sources.get(source, {}).get(page, "")):
            continue
        key = (source, page, normalized)
        if key not in seen:
            verified.append({"source_document": source, "page_number": page, "exact_quote": quote})
            seen.add(key)
    return verified


def normalize_asia_result(raw, target_countries, sources, complete_text=True):
    """Missing/malformed evidence becomes unknown, never zero-risk or automatic silence.

    Quote matching verifies provenance only. Relevance, scope and status interpretation
    are model-derived and still require human review.
    """
    rows = raw.get("country_assessments", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Provider returned no country assessments; retry with a structured-output model")
    indexed = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("country_code"), str):
            code = row["country_code"].upper()
            if code in target_countries:
                # Conflicting duplicate rows cannot be silently resolved as compliance.
                indexed[code] = row if code not in indexed else {}
    if not indexed:
        raise ValueError("Provider returned no requested country assessments")
    assessments = []
    warnings = []
    for code in target_countries:
        row = indexed.get(code, {})
        evidence = verify_evidence(row.get("evidence"), sources)
        operation_evidence = verify_evidence(row.get("operation_evidence"), sources)
        operations = row.get("operations_in_scope")
        operations = operations if type(operations) is bool and operation_evidence else None
        raw_status = row.get("status")
        status = raw_status if isinstance(raw_status, str) and raw_status in STATUS_POINTS else "unknown"
        if status == "silent":
            # Silence requires evidenced operations + evidenced global commitment,
            # a complete text extraction and an explicit description of the search.
            global_evidence = verify_evidence(row.get("global_commitment_evidence"), sources)
            checked = row.get("sections_checked")
            checked = [x for x in checked if isinstance(x, str) and x.strip()] if isinstance(checked, list) else []
            if operations is not True or not global_evidence or not checked or not complete_text:
                status = "unknown"
            evidence = global_evidence
        else:
            checked = []
            if not evidence:
                status = "unknown"
        if operations is False:
            status = "not_applicable"
        breakdown = []
        if status in STATUS_POINTS:
            breakdown.append({"factor": "status:" + status, "points": STATUS_POINTS[status]})
        accepted_patterns = []
        patterns = row.get("patterns", [])
        if isinstance(patterns, list) and status in STATUS_POINTS:
            for pattern in patterns:
                if not isinstance(pattern, dict):
                    continue
                name = pattern.get("finding_type")
                if not isinstance(name, str) or name not in PATTERN_POINTS or name in accepted_patterns:
                    continue
                proof = verify_evidence(pattern.get("evidence"), sources)
                # Delisting/downgrade require at least two distinct cited passages.
                if not proof or (name in {"silent_delisting", "commitment_downgrade"} and len(proof) < 2):
                    continue
                # Exclusion/deferral already counted by status: do not double charge.
                if (name == "geographic_exclusion" and status == "excluded") or (name == "timeline_deferral" and status == "deferred"):
                    continue
                accepted_patterns.append(name)
                breakdown.append({"factor": name, "points": PATTERN_POINTS[name], "evidence": proof})
        score = min(100, sum(x["points"] for x in breakdown)) if breakdown else None
        if status == "unknown":
            warnings.append(f"{code}: insufficient or unverified country evidence; no risk score assigned")
        assessments.append({
            "country_code": code, "country_name": COUNTRIES[code]["name"],
            "subregion": COUNTRIES[code]["subregion"], "status": status,
            "mentioned": bool(evidence) and status not in {"silent", "unknown", "not_applicable"},
            "operations_in_scope": operations, "operation_evidence": operation_evidence,
            "evidence": evidence, "sections_checked": checked,
            "risk_score": score, "risk_level": risk_level(score),
            "scoring_breakdown": breakdown,
            "explanation": str(row.get("explanation") or "No country-specific assessment supplied."),
        })
    scored = [x for x in assessments if x["risk_score"] is not None]
    regional_score = max((x["risk_score"] for x in scored), default=None)
    return {
        "schema_version": "3.0", "analysis_scope": "asia", "target_countries": target_countries,
        "country_catalog_version": CATALOG_VERSION, "regional_scoring_version": SCORING_VERSION,
        "country_assessments": assessments,
        "regional_risk_score": regional_score, "regional_risk_level": risk_level(regional_score),
        "regional_aggregation": "maximum_assessed_country_score",
        "coverage": {"requested": len(assessments), "assessed": len(scored),
                     "unknown": sum(x["status"] == "unknown" for x in assessments),
                     "not_applicable": sum(x["status"] == "not_applicable" for x in assessments),
                     "complete_text": complete_text},
        "validation_warnings": warnings,
        "overall_risk_scope": "Indonesia (legacy model-calculated methodology; not regional)",
        "review_required": True,
        "methodology_note": "Document-based risk screening, not legal compliance or independently verified corporate performance. Quote provenance is checked; country relevance and omission claims need human review. Unknown countries are not low risk. Regional v1 scores are not comparable to legacy scores.",
    }


def needs_regional_escalation(result):
    return (result["regional_risk_score"] or 0) >= 56 or any(
        row["status"] in {"silent", "excluded", "deferred"} for row in result["country_assessments"]
    )
