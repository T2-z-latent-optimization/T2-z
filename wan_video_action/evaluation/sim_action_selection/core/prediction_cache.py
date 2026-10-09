from __future__ import annotations

import copy
import hashlib
import json

import numpy as np


def input_digest(value):
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, np.ndarray):
            digest.update(b"array")
            digest.update(str(item.dtype).encode())
            digest.update(json.dumps(item.shape).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            digest.update(str(len(item)).encode())
            for child in item:
                update(child)
        else:
            digest.update(type(item).__name__.encode())
            digest.update(json.dumps(item, sort_keys=True).encode())

    update(value)
    return digest.hexdigest()


class CasePredictionCache:
    def __init__(self, worker):
        self.worker = worker
        self.context_signature = None
        self.context = None
        self.source_decision_uid = None
        self.predictions = {}

    def call(self, method, argument=None):
        if method == "freeze_context":
            signature = input_digest({key: value for key, value in argument.items() if key != "decision_uid"})
            if self.context is None:
                result = self.worker.call(method, argument)
                self.context = copy.deepcopy(result)
                self.context_signature = signature
                self.source_decision_uid = argument.get("decision_uid")
                return result
            if signature != self.context_signature:
                raise RuntimeError("Support inputs changed between targets within the same case")
            result = copy.deepcopy(self.context)
            result["context_reused_across_targets"] = True
            result["context_source_decision_uid"] = self.source_decision_uid
            return result
        if method == "predict":
            if self.context is None:
                raise RuntimeError("Freeze the case context before predicting candidates")
            signature = input_digest(argument)
            if signature not in self.predictions:
                self.predictions[signature] = np.array(self.worker.call(method, argument), copy=True)
            return self.predictions[signature].copy()
        return self.worker.call(method, argument)

    def shutdown(self):
        self.worker.shutdown()
