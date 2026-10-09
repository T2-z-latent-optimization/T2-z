from __future__ import annotations

import os
from pathlib import Path
import pickle
import select
import struct
import subprocess
import time


def write_message(stream, value):
    payload = pickle.dumps(value, protocol=4)
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


def read_message(stream, deadline=None):
    def exact(size):
        chunks = []
        while size:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
                    raise TimeoutError("Worker response timed out; see its log")
            chunk = os.read(stream.fileno(), min(size, 1024 * 1024))
            if not chunk:
                raise EOFError("Worker exited or closed its protocol pipe")
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    size = struct.unpack("!Q", exact(8))[0]
    if size > 4 * 1024 ** 3:
        raise ValueError("Oversized worker message")
    # This is a private pipe to a child we launched, never a network service.
    return pickle.loads(exact(size))


class WorkerClient:
    def __init__(self, python, kind, initialization, log_path, timeout=1800):
        self.timeout = float(timeout)
        self.log = Path(log_path).open("ab", buffering=0)
        worker = Path(__file__).with_name("worker.py")
        root = Path(__file__).resolve().parents[4]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            [str(python), "-u", str(worker), kind], cwd=root, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0,
        )
        try:
            self.call("initialize", initialization)
        except BaseException:
            self.close()
            raise

    def call(self, method, argument=None):
        if self.process.poll() is not None:
            raise RuntimeError(f"Worker exited with status {self.process.returncode}")
        write_message(self.process.stdin, {"method": method, "argument": argument})
        response = read_message(self.process.stdout, time.monotonic() + self.timeout)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response.get("result")

    def close(self):
        if getattr(self, "process", None) is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
        if getattr(self, "log", None) is not None:
            self.log.close()

    def shutdown(self):
        try:
            if self.process.poll() is None:
                self.call("close")
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            self.close()


class RemoteSimulator:
    def __init__(self, python, initialization, log_path, timeout=1800):
        self.client = WorkerClient(python, "simulator", initialization, log_path, timeout=timeout)

    def __getattr__(self, method):
        def operation(argument=None):
            return self.client.call(method, argument)
        return operation

    def close(self):
        self.client.shutdown()
