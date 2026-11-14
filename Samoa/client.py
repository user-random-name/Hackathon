"""Persistent simulator client for Linux and Windows x86-64."""

import json
import operator
import os
import subprocess
from pathlib import Path


class ArenaClient:
    def __init__(self, executable=None):
        if os.name == "nt":
            name = "samoa-windows-x86_64.exe"
        elif os.name == "posix":
            name = "samoa-linux-x86_64"
        else:
            raise RuntimeError("This release requires Linux or Windows x86-64")
        path = (
            Path(executable) if executable else Path(__file__).resolve().parent / name
        )
        self.process = subprocess.Popen(
            [str(path.resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.metadata = self._request("hello")
        if self.metadata.get("protocol") != 3:
            self.close()
            raise RuntimeError("Unsupported engine protocol")

    def _request(self, request):
        self.process.stdin.write(request + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("The simulator exited without a response")
        response = json.loads(line)
        if "error" in response:
            raise ValueError(response["error"])
        return response

    def reset(self, arena, seed=None):
        try:
            arena = operator.index(arena)
        except TypeError as exc:
            raise TypeError("Arena index must be an integer") from exc
        if seed is None:
            return self._request(f"reset {arena}")
        if isinstance(seed, bool):
            raise TypeError("Seed must be an integer")
        try:
            seed = operator.index(seed)
        except TypeError as error:
            raise TypeError("Seed must be an integer") from error
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("Seed must fit an unsigned 32-bit integer")
        return self._request(f"reset {arena} {seed}")

    def step(self, action):
        if action is not None:
            if isinstance(action, bool):
                raise TypeError("Return an integer cell index or None to rest")
            try:
                action = operator.index(action)
            except TypeError as exc:
                raise TypeError("Return an integer cell index or None to rest") from exc
            if not 0 <= action < self.metadata["size"] ** 2:
                raise ValueError("Cell index is outside this board; use None to rest")
        return self._request(f"step {-1 if action is None else action}")

    def begin_evaluation(self):
        return self._request("eval-begin")

    def next_evaluation_game(self):
        return self._request("eval-next")

    def finish_evaluation(self):
        return self._request("eval-finish")

    def observe(self):
        return self._request("observe")

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write("quit\n")
                self.process.stdin.flush()
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
