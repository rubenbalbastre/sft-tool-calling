"""Lifecycle management for a local vLLM server."""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen


def quantization_arguments(quantization):
    """Translate optional in-flight quantization into vLLM arguments."""
    if quantization in {None, "none"}:
        return []
    if quantization == "bnb_4bit":
        return ["--quantization", "bitsandbytes"]
    raise ValueError("quantization must be 'none' or 'bnb_4bit'")


def resolve_model_and_adapter(model):
    """Return the base model and optional PEFT adapter metadata."""
    adapter_config = Path(model) / "adapter_config.json"
    if not adapter_config.is_file():
        return model, None, None

    config = json.loads(adapter_config.read_text(encoding="utf-8"))
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"Missing base_model_name_or_path in {adapter_config}")
    return base_model, str(Path(model).resolve()), int(config["r"])


def resolve_tool_call_parser(model, configured_parser):
    """Select the native parser for known model families."""
    if configured_parser != "auto":
        return configured_parser

    model_name = str(model).lower()
    config_path = Path(model) / "config.json"
    if config_path.is_file():
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        model_name += " " + " ".join(model_config.get("architectures", []))

    return "gemma4" if "gemma-4" in model_name or "gemma4" in model_name else "hermes"


class VLLMServer:
    """Start vLLM, wait for readiness, and stop it after evaluation."""

    def __init__(
        self, model, config_path, base_url, timeout, log_path,
        served_model_name, tool_call_parser, quantization=None, adapter_path=None,
        adapter_rank=None, seed=0,
    ):
        self.model = model
        self.config_path = config_path
        self.base_url = base_url
        self.timeout = timeout
        self.log_path = log_path
        self.served_model_name = served_model_name
        self.tool_call_parser = resolve_tool_call_parser(model, tool_call_parser)
        self.quantization = quantization
        self.adapter_path = adapter_path
        self.adapter_rank = adapter_rank
        self.seed = seed
        self.process = None
        self.log_file = None

    def start(self):
        venv_executable = Path(sys.executable).with_name("vllm")
        executable = (
            str(venv_executable)
            if venv_executable.is_file()
            else shutil.which("vllm")
        )
        if not executable:
            raise RuntimeError(
                f"vLLM is not installed for {sys.executable}. "
                "Install the project requirements in this environment."
            )

        self.log_file = self.log_path.open("w", encoding="utf-8")
        command = [
            executable,
            "serve",
            self.model,
            "--config",
            str(self.config_path),
            "--served-model-name",
            (
                f"{self.served_model_name}-base"
                if self.adapter_path
                else self.served_model_name
            ),
            "--tool-call-parser",
            self.tool_call_parser,
            "--seed",
            str(self.seed),
        ]
        if self.tool_call_parser == "gemma4":
            command.extend(["--reasoning-parser", "gemma4"])
        command.extend(quantization_arguments(self.quantization))
        if self.adapter_path:
            command.extend([
                "--enable-lora",
                "--lora-modules",
                f"{self.served_model_name}={self.adapter_path}",
                "--max-lora-rank",
                str(self.adapter_rank),
            ])

        self.process = subprocess.Popen(
            command,
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
