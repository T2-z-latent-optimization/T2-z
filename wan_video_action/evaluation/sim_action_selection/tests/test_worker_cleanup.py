import subprocess
from unittest.mock import Mock

import pytest

from wan_video_action.evaluation.sim_action_selection.core.rpc import WorkerClient


def test_acknowledged_shutdown_reaps_slow_worker():
    client = WorkerClient.__new__(WorkerClient)
    client.process = Mock()
    client.process.poll.return_value = None
    client.process.wait.side_effect = [subprocess.TimeoutExpired("worker", 15), 0]
    client.call = Mock()
    client.close = Mock()
    client.shutdown()
    client.call.assert_called_once_with("close")
    client.process.kill.assert_called_once_with()
    client.close.assert_called_once_with()


def test_shutdown_preserves_worker_failure():
    client = WorkerClient.__new__(WorkerClient)
    client.process = Mock()
    client.process.poll.return_value = None
    client.call = Mock(side_effect=RuntimeError("original worker failure"))
    client.close = Mock()
    with pytest.raises(RuntimeError, match="original worker failure"):
        client.shutdown()
    client.close.assert_called_once_with()
