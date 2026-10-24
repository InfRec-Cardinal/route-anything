"""Shared routing policy for frozen capability predictions and live inference."""

UNRESOLVED_CLASS_ID = "__unresolved__"
FABLE = "claude-fable-5-1"
OPUS = "claude-opus-5-5"
FIXED_BENCHMARK_OVERRIDES = {}


def fixed_benchmark_model(table, benchmark):
    """Use the benchmark arm selected by observed training mean."""
    return FIXED_BENCHMARK_OVERRIDES.get(
        benchmark, table.get("benchmark_to_model", {}).get(benchmark))


def resolve(model, scores):
    """Map retired historical router choices to Opus, as in the revised paper."""
    redirected = model == FABLE
    effective = OPUS if redirected else model
    return effective, scores.get(effective), redirected


def route_prediction(prediction, data, table):
    """Choose the backend without modifying the classifier's saved output."""
    scores = data.get("model_scores", {})
    fixed = fixed_benchmark_model(table, data["benchmark"])
    unclassified = (
        prediction.get("predicted_leaf_id") is None
        and prediction.get("predicted_class_id") in (None, UNRESOLVED_CLASS_ID)
    )
    requested = (fixed if unclassified
                 else table["leaf_to_model"].get(prediction["predicted_leaf_id"]))
    model, quality, redirected = resolve(requested, scores)
    fixed_model, fixed_quality, fixed_redirected = resolve(fixed, scores)
    observed = [value for value in scores.values() if value is not None]
    return {
        "requested_model": requested,
        "selected_model": model,
        "unresolved_benchmark_fallback": unclassified,
        "fable_missing_redirected": redirected,
        "fable_missing_rescued": redirected and quality is not None,
        "router_quality": quality,
        "requested_fixed_model": fixed,
        "fixed_model": fixed_model,
        "fixed_fable_missing_redirected": fixed_redirected,
        "fixed_quality": fixed_quality,
        "oracle_quality": max(observed) if observed else None,
    }
