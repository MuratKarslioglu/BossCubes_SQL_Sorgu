"""Shared generation path for interactive requests and batched benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.errors import InvalidInputError, ModelInferenceError


@dataclass
class GenerationTiming:
    latency_ms: float
    prompt_tokens: int
    generated_tokens: int
    tokenize_ms: float = 0.0
    decode_ms: float = 0.0
    parse_ms: float = 0.0
    total_ms: float = 0.0
    first_token_ms: float | None = None


def synchronize(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class _FirstTokenTimer:
    """Optional profiler: streamer copies tokens to CPU, so keep it opt-in."""

    def __init__(self) -> None:
        self.started = 0.0
        self.first_token_ms = None
        self._prompt_seen = False

    def put(self, value) -> None:
        if not self._prompt_seen:
            self._prompt_seen = True
        elif self.first_token_ms is None:
            self.first_token_ms = (time.perf_counter() - self.started) * 1000

    def end(self) -> None:
        pass


def generate_prompts(
    loaded,
    prompts: list[str],
    max_new_tokens: int,
    *,
    max_input_tokens: int | None = None,
    profile_tokens: bool = False,
) -> list[tuple[str, GenerationTiming]]:
    import torch

    if not prompts:
        raise InvalidInputError("en az bir prompt gereklidir")
    total_start = time.perf_counter()
    tokenizer = loaded.tokenizer
    batched = len(prompts) > 1
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    try:
        if batched:
            tokenizer.padding_side = "left"
        kwargs = {"return_tensors": "pt"}
        if batched:
            kwargs["padding"] = True
        inputs = tokenizer(prompts if batched else prompts[0], **kwargs).to(loaded.device)
    finally:
        if batched:
            tokenizer.padding_side = original_padding_side
    input_len = inputs["input_ids"].shape[-1]
    if max_input_tokens is not None and input_len > max_input_tokens:
        raise InvalidInputError(f"girdi {max_input_tokens} token sinirini asiyor")
    synchronize(loaded.device)
    tokenize_ms = (time.perf_counter() - total_start) * 1000

    profiler = _FirstTokenTimer() if profile_tokens and not batched else None
    generation_kwargs = {"streamer": profiler} if profiler else {}
    start = time.perf_counter()
    if profiler:
        profiler.started = start
    try:
        with torch.inference_mode():
            output_ids = loaded.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                **generation_kwargs,
            )
        synchronize(loaded.device)
    except Exception as exc:
        raise ModelInferenceError("model cikarimi basarisiz oldu") from exc
    latency_ms = (time.perf_counter() - start) * 1000
    decode_start = time.perf_counter()
    results = []
    for i in range(len(prompts)):
        generated_ids = output_ids[i][input_len:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = (
            int((generated_ids != tokenizer.pad_token_id).sum())
            if batched
            else int(generated_ids.shape[-1])
        )
        prompt_tokens = int(inputs["attention_mask"][i].sum()) if batched else input_len
        results.append(
            (
                raw_text,
                GenerationTiming(
                    latency_ms=latency_ms / len(prompts),
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    tokenize_ms=tokenize_ms / len(prompts),
                    first_token_ms=profiler.first_token_ms if profiler else None,
                ),
            )
        )
    decode_ms = (time.perf_counter() - decode_start) * 1000 / len(prompts)
    total_ms = (time.perf_counter() - total_start) * 1000 / len(prompts)
    for _, timing in results:
        timing.decode_ms = decode_ms
        timing.total_ms = total_ms
    return results
