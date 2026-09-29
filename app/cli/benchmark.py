#!/usr/bin/env python3
"""Performans ve dogruluk benchmarki (spec §23).

    python main.py benchmark \\
      --model_name <HF_MODEL_ID> \\
      --adapter_path ./model_adapter \\
      --test_file storage/datasets/sql_json_dataset_v2_professional/test.jsonl

Held-out test seti uzerinde calisir. Olculenler: model yukleme suresi,
ortalama/medyan/p95 gecikme, prompt/uretilen token sayisi, mumkunse tepe
bellek, valid JSON orani, exact match, semantic match (spec §23).

Sonuclar `<output_dir>/<model>_<YYYYMMDD>.json`'a yazilir.

Spec §17: sifir-retry ve tek-retry ayri ayri benchmarklanmalidir. Varsayilan
`--allow_retry` KAPALIDIR (ilk-deneme dogrulugu spec'in birincil metrigidir);
tek-retry kolunu olcmek icin `--allow_retry` ile ayrica calistirin.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from app.errors import AppError
from app.inference.parsing import ExtractionResult, parse_query_json
from app.normalization.text import normalize_whitespace
from app.paths import project_path
from training.dataset import load_examples
from training.evaluate import (
    EvaluationRecord,
    EvaluationSummary,
    evaluate_records,
    evaluate_rejection,
    group_by_family,
    group_by_noise,
    group_by_status,
    latency_stats,
    status_confusion_matrix,
    token_stats,
)


def _summaries_to_dict(groups: dict[str, EvaluationSummary]) -> dict[str, dict[str, object]]:
    return {key: summary.to_dict() for key, summary in groups.items()}


def dump_predictions(records: list[EvaluationRecord], path: Path) -> None:
    """Her kayit icin {input, expected, predicted} JSONL'ini yazar (kullanicilara
    "su cumleyi verdik, model su JSON'u uretti" gostermek icin)."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def sanitize_model_name(name: str) -> str:
    """Dosya adi olarak guvenli bir slug uretir (spec §23 ornegi: model_name_YYYYMMDD.json)."""
    return name.strip().replace("/", "__").replace(" ", "_")


def peak_memory_mb() -> float | None:
    """Mumkunse tepe bellek kullanimini MB olarak doner (spec §23: 'where practical').

    `resource` modulu Windows'ta yoktur; bu durumda None doner (spec §37:
    sessizce yanlis bir deger uretmek yerine acikca eksik birakilir).
    `ru_maxrss` Linux'ta KB, macOS/BSD'de byte birimindedir.
    """
    try:
        import resource
    except ImportError:
        return None

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 if platform.system() == "Linux" else 1024 * 1024
    return raw / divisor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test seti uzerinde performans/dogruluk benchmarki (spec §23)."
    )
    parser.add_argument("--model_name", required=True, help="HuggingFace model id veya yerel yol")
    parser.add_argument("--adapter_path", default=None, type=Path)
    parser.add_argument("--test_file", required=True, type=Path)
    parser.add_argument("--output_dir", default=project_path("artifacts/benchmarks"), type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument("--prompt_format", choices=["auto", "plain", "chat"], default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--attention", choices=["auto", "eager", "sdpa"], default="auto")
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--profile_tokens", action="store_true")
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="--limit icin tekrar uretilebilir rastgele ornekleme",
    )
    parser.add_argument(
        "--allow_retry",
        action="store_true",
        help="spec §17'nin tek-retry kolunu olcer (varsayilan: sifir-retry / ilk-deneme)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="hizli deneme icin test setinden en fazla N ornek"
    )
    parser.add_argument(
        "--dump_predictions",
        type=Path,
        default=None,
        help="her ornek icin {input, expected, predicted} JSONL'ini bu yola yaz",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="1'den buyukse ornekler toplu (batched) generate() ile calisir -- "
        "Jetson gibi bellek bant genisligi kisitli cihazlarda buyuk test setlerini "
        "kat kat hizlandirir. Kisitlamalar: --allow_retry ile birlikte kullanilamaz "
        "(onarim mantigi tek ornek varsayar); rapordaki latency degerleri bu modda "
        "GERCEK tek-istek gecikmesi DEGIL, batch-amortize edilmis bir yaklasimdir "
        "(uretim gecikmesi icin --batch_size 1 kullanin).",
    )
    return parser


def build_report(
    args: argparse.Namespace,
    records: list[EvaluationRecord],
    model_load_seconds: float,
    resolved_device: str,
    resolved_dtype: str,
) -> dict[str, object]:
    """Saf rapor birlestirme (torch gerektirmez, ayri test edilebilir).

    V2 (DATASET_CARD.md): status/rejection/aile/gurultu bazinda kirilim da
    rapora dahil edilir. Bir dataset'te `status`/`metadata.family`/
    `metadata.noise` yoksa (V1 dataset) ilgili boyutlar bos sozluk olarak
    kalir -- hicbir sey uydurulmaz.
    """
    summary = evaluate_records(records)
    return {
        "model_name": args.model_name,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "device": resolved_device,
        "dtype": resolved_dtype,
        "retry_enabled": args.allow_retry,
        "prompt_format": args.prompt_format,
        "test_file": str(args.test_file),
        "test_examples": len(records),
        "model_load_seconds": model_load_seconds,
        "latency": latency_stats(records),
        "tokens": token_stats(records),
        "peak_memory_mb": peak_memory_mb(),
        "generated_at": datetime.now(UTC).isoformat(),
        **summary.to_dict(),
        "rejection": evaluate_rejection(records).to_dict(),
        "status_confusion_matrix": status_confusion_matrix(records),
        "by_status": _summaries_to_dict(group_by_status(records)),
        "by_family": _summaries_to_dict(group_by_family(records)),
        "by_noise": _summaries_to_dict(group_by_noise(records)),
    }


def _generate_batch(loaded, formatter, texts: list[str], max_new_tokens: int):
    """`texts` icin toplu (batched) generate() -- extract()'in tek-ornek yolunun
    aksine, GPU'yu N ornek uzerinde paralel calistirarak bellek bant genisligi
    darbogazini (Jetson gibi cihazlarda) amortize eder.

    Donen `GenerationTiming.latency_ms` TUM batch'in suresidir/N -- gercek
    tek-istek gecikmesi degil, ortalama bir yaklasimdir (bkz. --batch_size yardimi).
    """
    from app.inference.generation import generate_prompts

    return generate_prompts(loaded, [formatter.format_inference(t) for t in texts], max_new_tokens)


def run_batched(examples, loaded, formatter, args: argparse.Namespace) -> list[EvaluationRecord]:
    """--batch_size > 1 icin ana dongu; extract()'in normalize+parse mantigini
    (retry HARIC -- bkz. build_parser --batch_size yardimi) tekrarlar.

    `load_examples` her `RawExample.input`i yukleme sirasinda zaten
    `normalize_whitespace` sonrasi bos olmadigini dogruladigindan (bkz.
    `training/dataset.py::RawExample._validate_input`), extract()'in
    aksine burada ayrica bir "bos girdi" dalina gerek yoktur -- yine de
    tek-ornek yoluyla AYNI ciktiyi almak icin metin generate'ten once
    normalize edilir.
    """
    records: list[EvaluationRecord] = []
    progress = tqdm(
        range(0, len(examples), args.batch_size),
        desc=f"Benchmark [batch={args.batch_size}] [{args.test_file.name}]",
        unit="batch",
    )
    for start_idx in progress:
        chunk = examples[start_idx : start_idx + args.batch_size]
        texts = [normalize_whitespace(ex.input) for ex in chunk]

        generated = _generate_batch(loaded, formatter, texts, args.max_new_tokens)
        for ex, (raw_text, timing) in zip(chunk, generated, strict=True):
            try:
                result = ExtractionResult.ok(parse_query_json(raw_text))
            except AppError as exc:
                result = ExtractionResult.fail(exc)
            records.append(
                EvaluationRecord(
                    input=ex.input,
                    expected=ex.output,
                    result=result,
                    latency_ms=timing.latency_ms,
                    prompt_tokens=timing.prompt_tokens,
                    generated_tokens=timing.generated_tokens,
                    metadata=ex.metadata,
                )
            )
    return records


def print_compact_summary(report: dict[str, object]) -> None:
    """`by_status`/`by_family`/`by_noise` gibi genis kirilimlari EKRANA basmaz
    (bunlar zaten tam haliyle diskteki JSON dosyasinda duruyor) -- yalnizca
    spec §23'un baslik metriklerini ozetler."""
    latency = report["latency"]
    print("\nSonuclar:")
    print(f"  Ornek sayisi         : {report['test_examples']}")
    print(f"  Valid JSON orani     : {report['valid_json_rate']:.1%}")
    print(f"  Sema gecerlilik orani: {report['schema_validity_rate']:.1%}")
    print(f"  Exact match          : {report['exact_match_rate']:.1%}")
    print(f"  Semantic match       : {report['semantic_match_rate']:.1%}")
    print(
        "  Gecikme (ort/medyan/p95): "
        f"{latency['mean_ms']:.0f} / {latency['median_ms']:.0f} / {latency['p95_ms']:.0f} ms"
    )


def _run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.batch_size < 1
        or args.warmup_runs < 0
        or args.max_new_tokens < 1
        or (args.limit is not None and args.limit < 0)
    ):
        print(
            "HATA: batch/token degerleri pozitif, warmup_runs/limit negatif olmayan deger olmali.",
            file=sys.stderr,
        )
        return 1
    if args.profile_tokens and args.batch_size != 1:
        print("HATA: --profile_tokens yalnizca batch_size=1 icin kullanilir.", file=sys.stderr)
        return 1
    if args.adapter_path and not args.adapter_path.exists():
        print(f"HATA: adapter yolu bulunamadi: {args.adapter_path}", file=sys.stderr)
        return 1

    test_result = load_examples(args.test_file)
    if not test_result.ok:
        details = "\n".join(f"  {issue.format()}" for issue in test_result.issues)
        print(f"HATA: test dosyasi dogrulamadan gecemedi:\n{details}", file=sys.stderr)
        return 1

    examples = test_result.examples
    if args.limit is not None:
        if args.sample_seed is not None and 0 < args.limit < len(examples):
            examples = random.Random(args.sample_seed).sample(examples, args.limit)
        else:
            examples = examples[: args.limit]
    if not examples:
        print("HATA: benchmark icin ornek kalmadi (bos dosya veya --limit 0)", file=sys.stderr)
        return 1

    if args.batch_size > 1 and args.allow_retry:
        print(
            "HATA: --batch_size > 1, --allow_retry ile birlikte desteklenmiyor "
            "(onarim mantigi tek ornek varsayar); tek-retry kolu icin --batch_size 1 kullanin.",
            file=sys.stderr,
        )
        return 1

    # Gec import: --help ve dogrulama hatalari torch'u zorunlu kilmaz.
    from app.inference.extractor import extract
    from app.inference.model_loader import load_adapter, load_base_model, prepare_for_inference
    from training.formatting import get_formatter, resolve_prompt_format

    load_start = time.perf_counter()
    loaded = load_base_model(
        args.model_name,
        device=args.device,
        dtype=args.dtype,
        attention_implementation=args.attention,
    )
    if args.adapter_path:
        loaded.model = load_adapter(loaded.model, args.adapter_path)
    prepare_for_inference(loaded, merge_adapter=args.merge_adapter)
    model_load_seconds = time.perf_counter() - load_start

    prompt_format = resolve_prompt_format(args.prompt_format, loaded.tokenizer)
    formatter_kwargs = {"tokenizer": loaded.tokenizer} if prompt_format == "chat" else {}
    formatter = get_formatter(prompt_format, **formatter_kwargs)
    args.prompt_format = prompt_format
    cold_request_ms = None
    for _ in range(args.warmup_runs):
        warmup = extract(
            examples[0].input,
            loaded,
            formatter,
            max_new_tokens=args.max_new_tokens,
            allow_retry=args.allow_retry,
        )
        if cold_request_ms is None:
            cold_request_ms = warmup.timing.total_ms
    stage_timings = []

    if args.batch_size > 1:
        print(
            f"UYARI: --batch_size {args.batch_size} kullaniliyor -- rapordaki latency "
            "degerleri batch-amortize edilmistir, gercek tek-istek (uretim) gecikmesini "
            "yansitmaz.",
            file=sys.stderr,
        )
        records = run_batched(examples, loaded, formatter, args)
    else:
        records = []
        progress = tqdm(examples, desc=f"Benchmark [{args.test_file.name}]", unit="ornek")
        for example in progress:
            outcome = extract(
                example.input,
                loaded,
                formatter,
                max_new_tokens=args.max_new_tokens,
                allow_retry=args.allow_retry,
                profile_tokens=args.profile_tokens,
            )
            stage_timings.append(outcome.timing)
            records.append(
                EvaluationRecord(
                    input=example.input,
                    expected=example.output,
                    result=outcome.result,
                    latency_ms=outcome.timing.total_ms or outcome.timing.latency_ms,
                    prompt_tokens=outcome.timing.prompt_tokens,
                    generated_tokens=outcome.timing.generated_tokens,
                    metadata=example.metadata,
                )
            )

    report = build_report(args, records, model_load_seconds, loaded.device, str(loaded.dtype))
    report["batch_size"] = args.batch_size
    report["latency_kind"] = "single_request" if args.batch_size == 1 else "batch_amortized"
    report["merged_adapter"] = args.merge_adapter
    report["attention"] = args.attention
    report["max_new_tokens"] = args.max_new_tokens
    report["sample_seed"] = args.sample_seed
    report["cold_request_ms"] = cold_request_ms
    report["warmup_runs"] = args.warmup_runs
    report["profiling_enabled"] = args.profile_tokens
    if stage_timings:
        report["stage_mean_ms"] = {
            name: sum(getattr(t, name) or 0 for t in stage_timings) / len(stage_timings)
            for name in ("tokenize_ms", "latency_ms", "decode_ms", "parse_ms", "total_ms")
        }
        first = [t.first_token_ms for t in stage_timings if t.first_token_ms is not None]
        report["first_token_mean_ms"] = sum(first) / len(first) if first else None
    from app.inference.diagnostics import system_report

    report["environment"] = system_report()

    if args.dump_predictions is not None:
        args.dump_predictions.parent.mkdir(parents=True, exist_ok=True)
        dump_predictions(records, args.dump_predictions)
        print(f"Ornek bazli ciktilar yazildi: {args.dump_predictions}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.output_dir
        / f"{sanitize_model_name(args.model_name)}_{datetime.now(UTC):%Y%m%d_%H%M%S_%f}.json"
    )
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_compact_summary(report)
    print(f"\nTam rapor: {out_path}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except AppError as exc:
        print(f"HATA [{exc.error_type.value}]: {exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
