"""Project-wide candidate exclusions for newly prepared experiments.

Raw source evidence and already published historical results remain unchanged.
Candidate-letter SFT datasets must be prepared again after the pool changes.
"""
import re


EXCLUDED_MODEL_IDS = ["claude-fable-5-1"]
POLICY_VERSION = "exclude-fable-5.1-20261020"


def allowed(model_id):
    identity = str(model_id or "").split(" | ", 1)[0].split("/")[-1].lower()
    normalized = re.sub(r"[^a-z0-9]", "", identity)
    return normalized not in {"claudefable51", "fable51"}


def candidate_model(candidate):
    profile = candidate.get("profile") or {}
    return candidate.get("model_id") or profile.get("model_id") or profile.get("model")


def filter_record(row):
    """Filter score/candidate pools before fitting or oracle construction."""
    result = dict(row)
    for field in ("model_scores", "model_costs", "model_tokens", "model_arm_versions",
                  "model_repeat_count", "observations"):
        if field in result:
            result[field] = {name: value for name, value in result[field].items() if allowed(name)}
    if "candidate_models" in result:
        result["candidate_models"] = [name for name in result["candidate_models"] if allowed(name)]
    if "candidates" in result:
        result["candidates"] = [c for c in result["candidates"] if allowed(candidate_model(c))]
    if "model_scores" in result:
        count = len(result["model_scores"])
        for field, value in (("source_single_candidate", count == 1),
                             ("routing_comparison_available", count > 1),
                             ("candidate_model_count", count)):
            if field in result:
                result[field] = value
    result["model_exclusion_policy"] = POLICY_VERSION
    return result


def require_allowed_candidates(row):
    """Reject stale candidate-letter data; deleting letters at inference is invalid."""
    excluded = sorted({str(candidate_model(c)) for c in row.get("candidates", [])
                       if not allowed(candidate_model(c))})
    if excluded:
        raise ValueError(
            f"{row.get('decision_id', row.get('id', 'record'))} contains excluded models {excluded}. "
            "Re-run prepare into a new directory so candidate text, labels, baseline and oracle "
            "are rebuilt together; do not delete candidate letters at inference."
        )
    return row


def require_current_table(table):
    """A table fitted with the old candidate pool must be refitted on train."""
    if table.get("model_exclusion_policy") != POLICY_VERSION:
        raise ValueError(
            "This frozen routing table predates the Fable 5.1 exclusion. "
            "Re-run prepare/mapping with training data into a new directory; "
            "do not replace Fable after inspecting evaluation outcomes."
        )
    return table
