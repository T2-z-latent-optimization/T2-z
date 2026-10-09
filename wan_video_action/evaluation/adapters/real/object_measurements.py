"""Missing-aware measurements. Scores never silently interpolate absent objects."""
from __future__ import annotations

import numpy as np

OFFSETS = {"ball": (80,156), "door": (0,0), "stick": (0,0), "soft": (32,224)}


def usable(state):
    return (state.get("center") is not None and
            not state.get("ambiguous", False) and
            state.get("measurement_valid", True) and
            state.get("confidence",0.) >= .5)


def trajectory_metrics(gt, predicted, task, valid_frames):
    """Frame-aligned reconstruction only; exclude input/padded frames via mask.

    valid_frames must explicitly represent the common REAL prediction horizon.
    Counterfactual videos without matched GT must not call this function.
    """
    if not len(gt) == len(predicted) == len(valid_frames):
        raise ValueError("GT/prediction/frame masks must have identical lengths")
    wanted = np.flatnonzero(np.asarray(valid_frames,dtype=bool))
    pairs = [i for i in wanted if usable(gt[i]) and usable(predicted[i])]
    result = {"requested_frames": int(len(wanted)), "paired_frames": len(pairs),
              "coverage": len(pairs)/len(wanted) if len(wanted) else 0.,
              "gt_missing_frames": sum(not usable(gt[i]) for i in wanted),
              "prediction_missing_frames": sum(not usable(predicted[i]) for i in wanted),
              "center_ade_px": None, "center_fde_px": None,
              "center_ade_normalized": None, "complete_horizon": len(pairs)==len(wanted)
              and len(wanted)>0}
    if not pairs:
        return result
    errors = [np.linalg.norm(np.asarray(predicted[i]["center"])-
                             np.asarray(gt[i]["center"])) for i in pairs]
    result["center_ade_px"] = float(np.mean(errors))
    result["center_ade_normalized"] = result["center_ade_px"]/800.
    if pairs[-1] == wanted[-1]:
        result["center_fde_px"] = float(errors[-1])
    if task == "ball":
        result["observed_peak_x_error_px"] = abs(
            max(predicted[i]["center"][0] for i in pairs)-
            max(gt[i]["center"][0] for i in pairs))
        result["peak_error_certified_full_horizon"] = result["complete_horizon"]
    if task == "stick":
        angular = [abs((predicted[i]["angle_deg"]-gt[i]["angle_deg"]+90)%180-90)
                   for i in pairs if "angle_deg" in gt[i] and "angle_deg" in predicted[i]]
        result["angle_mae_deg"] = float(np.mean(angular)) if angular else None
    return result


def episode_diagnostics(task, states, fps, calibration):
    good = [s for s in states if usable(s)]
    result = {"usable_fraction": len(good)/len(states) if states else 0.,
              "edge_frames": sum(bool(s.get("touches_image_edge") or
                                     s.get("touches_right_edge")) for s in states),
              "formal_gold": False,
              "raw_ncc_ge_05_fraction": sum(s.get("center") is not None and
                    s.get("template_source") is not None and s.get("confidence",0)>=.5
                    for s in states)/len(states) if states else 0.,
              "flow_frames": sum(s.get("reason")=="anchored_rigid_flow" for s in states),
              "missing_frames": sum(s.get("center") is None for s in states)}
    if not good:
        return result
    jumps = [np.linalg.norm(np.asarray(b["center"])-np.asarray(a["center"]))
             for a,b in zip(states,states[1:]) if usable(a) and usable(b)]
    result["max_adjacent_jump_px"] = float(max(jumps)) if jumps else None
    if task == "soft":

        count=max(3,round(.2*fps))
        first,last=states[:count],states[-count:]
        if all(usable(s) for s in first+last):
            origin=np.median([s["center"] for s in first],axis=0)
            final=np.median([s["center"] for s in last],axis=0)
            result["sustained_dx_px"]=float(final[0]-origin[0])
            result["sustained_translation_px"]=float(np.linalg.norm(final-origin))
            result["left_excursion_px"]=float(max(0.,origin[0]-min(s["center"][0] for s in good)))
            result["right_excursion_px"]=float(max(0.,max(s["center"][0] for s in good)-origin[0]))
            result["direction"]=("left" if final[0]-origin[0] <= -15 else
                                 "right" if final[0]-origin[0] >= 15 else "small_or_rotation")
        result["support_motion_proposal"]=(
            result["usable_fraction"]>=.98 and result["edge_frames"]==0 and
            abs(result.get("sustained_dx_px",0))>=15 and
            max(result.get("left_excursion_px",0),result.get("right_excursion_px",0))>=25)
    elif task == "door":
        tail=states[-max(1,round(.3*fps)):]
        labels=[s.get("closure_state","uncertain") for s in tail]
        result["stable_closure"]=("uncertain" if not tail or "uncertain" in labels else
                                  "closed" if all(x=="closed" for x in labels) else "open_or_rebound")
        result["uncertain_closure_frames"]=sum(s.get("closure_state")=="uncertain" for s in states)
    elif task == "ball":
        result["observed_peak_original_x"]=max(s["center"][0] for s in good)+80
        result["near_target_boundary"]=min(abs(result["observed_peak_original_x"]-v)
                                          for v in (280,330))<=3
        result["peak_measurement_needs_review"]=(result["near_target_boundary"] or
                                                result["usable_fraction"]<1.)
    return result
