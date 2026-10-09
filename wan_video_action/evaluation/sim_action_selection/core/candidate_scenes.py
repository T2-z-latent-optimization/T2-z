from __future__ import annotations

import copy
import hashlib
import json

import numpy as np


class CandidateScenes:
    def __init__(self, factory, case):
        self.factory = factory
        self.case = copy.deepcopy(case)
        self.initializations = {int(index): copy.deepcopy(values)
                                for index, values in self.case.pop("candidate_initializations").items()}
        self.primary = factory(copy.deepcopy(self.case))
        self.reference = None
        self.prepared = {}
        self.actors = {}
        self.paused = {}
        self.executed = False

    def prepare(self):
        paused = self.primary.prepare()
        indices = {int(candidate["index"]) for candidate in self.primary.candidates()}
        if set(self.initializations) != indices:
            raise ValueError("Candidate initializations must cover the complete action bank")
        paused["candidate_scene_mode"] = "candidate_specific_scenes"
        paused["shared_candidate_first_frame"] = False
        return paused

    def candidates(self):
        return self.primary.candidates()

    def fingerprint(self):
        return self.primary.fingerprint()

    def set_candidate_actions(self, reference):
        if self.prepared or self.executed:
            raise RuntimeError("Install reference actions before preparing candidate scenes")
        result = self.primary.set_candidate_actions(reference)
        self.reference = {key: np.asarray(value).copy() for key, value in reference.items()}
        return result

    def _actor(self, index):
        if index not in self.initializations:
            raise KeyError(f"Unknown candidate scene: {index}")
        case = {**copy.deepcopy(self.case), **copy.deepcopy(self.initializations[index])}
        return self.factory(case)

    def _prepare(self, actor, index):
        paused = actor.prepare()
        if self.reference is not None:
            actor.set_candidate_actions(self.reference)
        candidate = next(item for item in actor.candidates() if int(item["index"]) == index)
        signature = {
            "fingerprint": paused["fingerprint"],
            "current_rgb_sha256": hashlib.sha256(np.ascontiguousarray(paused["views"]).tobytes()).hexdigest(),
            "model_actions_sha256": hashlib.sha256(np.ascontiguousarray(candidate["model_actions"]).tobytes()).hexdigest(),
            "native_frame_indices": np.asarray(paused["native_frame_indices"]).tolist(),
        }
        return paused, candidate, signature

    def prepare_candidate(self, action_index):
        if self.executed:
            raise RuntimeError("Candidate execution has already begun")
        index = int(action_index)
        if index in self.prepared:
            actor = self.actors[index]
            candidate = next(item for item in actor.candidates() if int(item["index"]) == index)
            return {"paused": copy.deepcopy(self.paused[index]), "candidate": candidate,
                    "signature": copy.deepcopy(self.prepared[index])}
        actor = self._actor(index)
        try:
            paused, candidate, signature = self._prepare(actor, index)
            self.prepared[index] = signature
            self.actors[index] = actor
            paused["candidate_scene_mode"] = "candidate_specific_scenes"
            paused["shared_candidate_first_frame"] = False
            self.paused[index] = copy.deepcopy(paused)
            return {"paused": paused, "candidate": candidate, "signature": signature}
        except BaseException:
            actor.close()
            raise

    def execute(self, action_index):
        if self.executed:
            raise RuntimeError("Candidate execution has already begun")
        index = int(action_index)
        if index not in self.prepared:
            self.prepare_candidate(index)
        actor = self.actors[index]
        try:
            candidate = next(item for item in actor.candidates() if int(item["index"]) == index)
            signature = {
                "fingerprint": actor.fingerprint(),
                "current_rgb_sha256": hashlib.sha256(np.ascontiguousarray(actor.current()["views"]).tobytes()).hexdigest(),
                "model_actions_sha256": hashlib.sha256(np.ascontiguousarray(candidate["model_actions"]).tobytes()).hexdigest(),
                "native_frame_indices": self.prepared[index]["native_frame_indices"],
            }
            if signature != self.prepared[index]:
                differences = {key: {"prepared": self.prepared[index][key], "reconstructed": value}
                               for key, value in signature.items() if value != self.prepared[index][key]}
                raise RuntimeError("Selected candidate scene or commands changed after prediction: " +
                                   json.dumps(differences, sort_keys=True))
            self.executed = True
            result = actor.execute(index)
            result.update({
                "candidate_scene_mode": "candidate_specific_scenes",
                "execution_initial_fingerprint": signature["fingerprint"],
                "execution_initial_rgb_sha256": signature["current_rgb_sha256"],
                "selected_scene_reproduced_after_prediction": True,
                "selected_scene_restoration": "retained_paused_instance",
            })
            return result
        finally:
            self.actors.pop(index, None)
            actor.close()

    def close(self):
        actors = list(self.actors.values())
        self.actors.clear()
        try:
            for actor in actors:
                actor.close()
        finally:
            self.primary.close()
