"""One model per process, shared by API and interactive CLI."""

from __future__ import annotations

import time
from dataclasses import asdict

from app.config import Settings
from app.errors import InvalidInputError, ModelInferenceError
from app.inference.extractor import ExtractionOutcome
from app.normalization.text import normalize_whitespace
from app.paths import model_reference, project_path


class InferenceEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.loaded = None
        self.formatter = None
        self.load_seconds = 0.0
        self.cold_request_ms = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        from app.inference.model_loader import (
            load_adapter,
            load_base_model,
            prepare_for_inference,
        )
        from training.formatting import get_formatter, resolve_prompt_format

        started = time.perf_counter()
        self.loaded = load_base_model(
            model_reference(self.settings.base_model),
            device=self.settings.device,
            dtype=self.settings.model_dtype,
            attention_implementation=self.settings.attention_implementation,
        )
        if self.settings.adapter_path:
            self.loaded.model = load_adapter(
                self.loaded.model, project_path(self.settings.adapter_path)
            )
        prepare_for_inference(self.loaded, merge_adapter=self.settings.merge_adapter)
        prompt_format = resolve_prompt_format(self.settings.prompt_format, self.loaded.tokenizer)
        kwargs = {"tokenizer": self.loaded.tokenizer} if prompt_format == "chat" else {}
        self.formatter = get_formatter(prompt_format, **kwargs)
        self.load_seconds = time.perf_counter() - started
        for _ in range(self.settings.warmup_runs):
            outcome = self.extract("ABC firmasının satış sayısını getir.")
            if self.cold_request_ms is None:
                self.cold_request_ms = outcome.timing.total_ms
        self._started = True

    def extract(self, text: str, *, profile_tokens: bool = False) -> ExtractionOutcome:
        text = normalize_whitespace(text)
        if not text or len(text) > self.settings.max_input_chars:
            raise InvalidInputError("girdi bos veya izin verilen uzunlugu asiyor")
        from app.inference.extractor import extract

        if self.loaded is None or self.formatter is None:
            raise ModelInferenceError("model baslatilmamis")

        return extract(
            text,
            self.loaded,
            self.formatter,
            max_new_tokens=self.settings.max_new_tokens,
            allow_retry=self.settings.allow_retry,
            max_input_tokens=self.settings.max_input_tokens,
            profile_tokens=profile_tokens,
        )

    def info(self) -> dict:
        return {
            "model": self.settings.base_model,
            "adapter": self.settings.adapter_path or None,
            "device": self.loaded.device if self.loaded else None,
            "dtype": str(self.loaded.dtype) if self.loaded else None,
            "runtime": "pytorch",
            "merged_adapter": bool(self.settings.adapter_path) and self.settings.merge_adapter,
            "prompt_format": self.formatter.name if self.formatter else None,
            "max_new_tokens": self.settings.max_new_tokens,
            "retry_enabled": self.settings.allow_retry,
            "model_load_seconds": self.load_seconds,
            "cold_request_ms": self.cold_request_ms,
        }


def outcome_payload(outcome: ExtractionOutcome) -> dict:
    return {
        **outcome.result.model_dump(mode="json"),
        "timing": asdict(outcome.timing),
        "retried": outcome.retried,
    }
