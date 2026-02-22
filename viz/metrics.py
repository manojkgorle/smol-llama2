"""Training metrics logger with thread-safe dual storage (in-memory + JSONL)."""

import json
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass


@dataclass
class StepMetrics:
    step: int
    epoch: int
    loss: float
    perplexity: float
    learning_rate: float
    grad_norm: float
    wall_time: float
    residual_norms: list[float] | None = None
    attn_output_norms: list[float] | None = None
    ffn_output_norms: list[float] | None = None
    ffn_gate_sparsity: list[float] | None = None
    per_layer_grad_norms: list[float] | None = None


@dataclass
class ValMetrics:
    step: int
    val_loss: float
    val_perplexity: float


class MetricsLogger:
    """Thread-safe metrics logger with in-memory deque and JSONL file storage."""

    def __init__(self, log_dir: str = "viz_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._steps: deque[dict] = deque()
        self._validations: deque[dict] = deque()

        self._train_path = os.path.join(log_dir, "train_metrics.jsonl")
        self._val_path = os.path.join(log_dir, "val_metrics.jsonl")

        self.load_from_file()

    def log_step(self, metrics: StepMetrics) -> None:
        record = asdict(metrics)
        with self._lock:
            self._steps.append(record)
            with open(self._train_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def log_validation(self, metrics: ValMetrics) -> None:
        record = asdict(metrics)
        with self._lock:
            self._validations.append(record)
            with open(self._val_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def get_all_steps(self) -> list[dict]:
        with self._lock:
            return list(self._steps)

    def get_all_validations(self) -> list[dict]:
        with self._lock:
            return list(self._validations)

    def get_steps_since(self, step: int) -> list[dict]:
        with self._lock:
            return [s for s in self._steps if s["step"] > step]

    def load_from_file(self) -> None:
        with self._lock:
            self._steps.clear()
            self._validations.clear()

            if os.path.exists(self._train_path):
                with open(self._train_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._steps.append(json.loads(line))

            if os.path.exists(self._val_path):
                with open(self._val_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._validations.append(json.loads(line))
