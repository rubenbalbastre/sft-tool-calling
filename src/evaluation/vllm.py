"""Lifecycle management for a local vLLM server."""

import shutil
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import urlopen


class VLLMServer:
    """Start vLLM, wait for readiness, and stop it after evaluation."""

    def __init__(self, model, config_path, base_url, timeout, log_path):
        self.model = model
        self.config_path = config_path
        self.base_url = base_url
        self.timeout = timeout
        self.log_path = log_path
        self.process = None
        self.log_file = None

    def start(self):
        executable = shutil.which("vllm")
        if not executable:
            raise RuntimeError("vllm executable not found in the active environment")

        self.log_file = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                executable,
                "serve",
                self.model,
                "--config",
                str(self.config_path),
            ],
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise

    def _wait_until_ready(self):
        parsed = urlsplit(self.base_url)
        health_url = f"{parsed.scheme}://{parsed.netloc}/health"
        deadline = time.monotonic() + self.timeout

        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup; see {self.log_path}"
                )
            try:
                with urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(1)

        raise TimeoutError(
            f"vLLM was not ready after {self.timeout} seconds; "
            f"see {self.log_path}"
        )

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log_file:
            self.log_file.close()
