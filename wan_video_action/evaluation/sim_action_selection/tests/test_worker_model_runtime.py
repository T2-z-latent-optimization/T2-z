import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from wan_video_action.evaluation.sim_action_selection.core.worker import configure_model_runtime


class ModelRuntimeTests(unittest.TestCase):
    def test_default_does_not_import_or_configure_torch(self):
        with patch.dict(sys.modules, {"torch": None}), patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(configure_model_runtime({}))
            self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", os.environ)

    def test_explicit_false_preserves_environment(self):
        with patch.dict(sys.modules, {"torch": None}), patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "existing"}):
            self.assertIsNone(configure_model_runtime({"overrides": {"deterministic_algorithms": False}}))
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], "existing")

    def test_enabled_configures_child_without_changing_precision(self):
        calls = []
        precision = object()
        cudnn = SimpleNamespace(deterministic=False, benchmark=True, allow_tf32=precision)
        matmul = SimpleNamespace(allow_tf32=precision)
        fake = SimpleNamespace(
            use_deterministic_algorithms=Mock(side_effect=lambda enabled: calls.append((enabled, os.environ.get("CUBLAS_WORKSPACE_CONFIG")))),
            are_deterministic_algorithms_enabled=lambda: bool(calls and calls[-1][0]),
            backends=SimpleNamespace(cudnn=cudnn, cuda=SimpleNamespace(matmul=matmul)),
        )
        with patch.dict(sys.modules, {"torch": fake}), patch.dict(os.environ, {}, clear=True):
            report = configure_model_runtime({"overrides": {"deterministic_algorithms": True}})
        self.assertEqual(calls, [(True, ":4096:8")])
        self.assertEqual(report, {"deterministic_algorithms": True, "cublas_workspace_config": ":4096:8", "cudnn_deterministic": True, "cudnn_benchmark": False})
        self.assertIs(cudnn.allow_tf32, precision)
        self.assertIs(matmul.allow_tf32, precision)


if __name__ == "__main__":
    unittest.main()
