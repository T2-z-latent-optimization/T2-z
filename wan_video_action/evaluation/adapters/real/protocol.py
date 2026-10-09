"""Frozen eligibility and support-quality policy; never reads query outcomes."""

def ball_support_eligibility(record):
    if not record.get("training_eligible", False):
        return False, "not_eligible_train"
    if record.get("missing_rate", 1.0) > .02 or record.get("ambiguous_frames", 1) > 0:
        return False, "tracking_requires_review"
    peak = record.get("peak_x_original_px")
    if peak is None:
        return False, "no_ball_peak"
    if 275 <= peak <= 335:
        return False, "target_or_target_boundary"
    if peak > 580 or record.get("boundary_seen"):
        return False, "too_far_or_boundary"
    if record.get("post_action_roll_px", 0) < 25:
        return False, "insufficient_free_roll"
    return True, "candidate_requires_chunk_and_visual_audit"


def support_source_allowed(row):
    return row["split"] == "train"
