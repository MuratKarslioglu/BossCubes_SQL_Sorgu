#!/usr/bin/env python3
"""Tek komutluk egitim + benchmark orkestratoru (su an run_004_jetson icin).

`training/train.py` ve `scripts/benchmark.py` zaten test edilmis `main(argv)`
giris noktalarini sirayla cagirir -- egitim/degerlendirme mantigi burada
TEKRAR YAZILMAZ. Bu betik yalnizca bu belirli run icin sabit degerlerle
orkestrasyon yapar; parametrik hale getirmek (argparse vb.) su an icin
istenmeyen bir genelleme.

Kullanim:
    python main.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import benchmark  # noqa: E402

from training import train  # noqa: E402

# train.main() kendi logging.basicConfig(INFO) cagrisini yapiyor (spec §26);
# bu, root logger'a baglanan huggingface_hub/httpx/urllib3 gibi kutuphanelerin
# de INFO seviyesinde (HEAD/GET istek loglari) konusmasina yol aciyor. Bu
# pipeline'in kendi ilerleme/sonuc ciktisi disinda gurultu istemedigimiz
# icin yalnizca bu bilinen gurultulu logger'lari WARNING'e cekiyoruz --
# train.py/model_loader.py'nin kendi anlamli INFO loglarina dokunulmuyor.
for _noisy_logger in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

RUN_NAME = "run_005_jetson"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# run_003'te kullanilan "datasets/external/train_synthetic_plus_external.jsonl"
# (12k sentetik + TUR2SQL/TURSpider'dan 9k gercek, toplam 21k) bu makinede
# YOK -- o dosya ve onu ureten datasets/external/ hic git'e commit edilmemis,
# yalnizca calismanin yapildigi Mac'te yerel kalmis (git log --all bunu
# dogruluyor). Elimizdeki tek egitim verisi 12k'lik sentetik V2 seti.
TRAIN_FILE = "datasets/sql_json_dataset_v2_professional/train.jsonl"
VALIDATION_FILE = "datasets/sql_json_dataset_v2_professional/validation.jsonl"
MODEL_OUTPUT_DIR = f"models/{RUN_NAME}"

# Hiperparametreler run_002/run_003 ile birebir ayni (spec'in onerdigi
# baslangic degerleri, Mac'teki run_003'te train_loss=0.011'e yakinsamisti).
EPOCHS = "3"
LEARNING_RATE = "2e-4"
BATCH_SIZE = "4"
GRADIENT_ACCUMULATION_STEPS = "4"
LORA_R = "16"
LORA_ALPHA = "32"
LORA_DROPOUT = "0.05"
PROMPT_FORMAT = "chat"

BENCHMARK_TEST_FILES = {
    "test": "datasets/sql_json_dataset_v2_professional/test.jsonl",
    "challenge_test": "datasets/sql_json_dataset_v2_professional/challenge_test.jsonl",
    # Kaggle "turkish-query-sql" setinden donusturulen 136 gercek-dunya
    # Turkce ornegi (bkz. scripts/import_kaggle_turkish_sql.py) -- ilk
    # gercek OOD degerlendirmemiz. run_003'teki 3 OOD dosyasi
    # (tur2sql_ood, turspider_ood, turkish_real_nl) hala bu makinede YOK;
    # TUR2SQL/TURSpider ham verisi indirilip
    # scripts/import_external_sql_dataset.py --ood-out-dir ile uretilene
    # kadar cikarildi.
    "kaggle_real_nl_ood": "datasets/ood_eval/kaggle_turkish_real_nl_ood.jsonl",
}

# Jetson'da kurulu torch derlemesi Orin'in CUDA compute capability'sini
# (8.7) resmi desteklemiyor (bkz. run_004_jetson_pipeline.log'daki
# "No published PyTorch CUDA builds ... support this GPU" uyarisi) --
# tek-ornek (batch=1) uretim bu yuzden bellek bant genisligi tarafindan
# agir sekilde darbogazlaniyor. Bu Jetson'a ozgu makinede olcumlenmistir:
# batch=64 ile tam test seti benchmark'i ~13 saatten ~25 dakikaya indi
# (ayni exact/semantic match oranlariyla). Rapordaki latency degerleri bu
# modda batch-amortize edilmis olur, gercek tek-istek uretim gecikmesini
# YANSITMAZ -- o degeri olcmek icin ayrica --batch_size 1 ile calistirin.
BENCHMARK_BATCH_SIZE = "64"


TOTAL_STAGES = 1 + len(BENCHMARK_TEST_FILES)  # egitim + her benchmark dosyasi
_stage_counter = 0


def _stage_header(title: str) -> None:
    """Genel pipeline ilerlemesi: 'ADIM k/TOTAL' -- ne kadar kaldigini gosterir."""
    global _stage_counter
    _stage_counter += 1
    print("\n" + "=" * 70)
    print(f"ADIM {_stage_counter}/{TOTAL_STAGES} -- {title}")
    print("=" * 70, flush=True)


def run_training() -> tuple[int, float]:
    _stage_header(f"EGITIM ({MODEL_NAME}, {EPOCHS} epoch)")
    argv = [
        "--model_name", MODEL_NAME,
        "--train_file", TRAIN_FILE,
        "--validation_file", VALIDATION_FILE,
        "--output_dir", MODEL_OUTPUT_DIR,
        "--epochs", EPOCHS,
        "--learning_rate", LEARNING_RATE,
        "--batch_size", BATCH_SIZE,
        "--gradient_accumulation_steps", GRADIENT_ACCUMULATION_STEPS,
        "--lora_r", LORA_R,
        "--lora_alpha", LORA_ALPHA,
        "--lora_dropout", LORA_DROPOUT,
        "--prompt_format", PROMPT_FORMAT,
    ]  # fmt: skip
    start = time.perf_counter()
    exit_code = train.main(argv)
    duration = time.perf_counter() - start
    status = "TAMAMLANDI" if exit_code == 0 else "BASARISIZ"
    print(f"\n-- EGITIM {status} ({duration:.0f}s) --", flush=True)
    return exit_code, duration


def run_benchmarks() -> dict[str, dict[str, object]]:
    """Her test dosyasi icin ayri output_dir kullanir -- ayni gun/model icin
    benchmark.py'nin `<model>_<YYYYMMDD>.json` dosya adi cakismasin diye."""
    results: dict[str, dict[str, object]] = {}
    for label, test_file in BENCHMARK_TEST_FILES.items():
        _stage_header(f"BENCHMARK: {label} ({test_file})")
        output_dir = f"benchmarks/{RUN_NAME}/{label}"
        argv = [
            "--model_name", MODEL_NAME,
            "--adapter_path", MODEL_OUTPUT_DIR,
            "--test_file", test_file,
            "--output_dir", output_dir,
            "--dump_predictions", f"{output_dir}/predictions.jsonl",
            "--batch_size", BENCHMARK_BATCH_SIZE,
        ]  # fmt: skip
        start = time.perf_counter()
        try:
            exit_code = benchmark.main(argv)
        except Exception as exc:  # bir test setinin patlamasi digerlerini durdurmasin
            print(f"HATA: {label} benchmark'i patladi: {exc}", file=sys.stderr)
            exit_code = 1
        duration = time.perf_counter() - start
        status = "TAMAMLANDI" if exit_code == 0 else "BASARISIZ"
        print(f"\n-- BENCHMARK [{label}] {status} ({duration:.0f}s) --", flush=True)
        results[label] = {
            "test_file": test_file,
            "output_dir": output_dir,
            "exit_code": exit_code,
            "duration_seconds": duration,
        }
    return results


def print_summary(training: dict[str, object], benchmarks: dict[str, dict[str, object]]) -> None:
    print("\n" + "=" * 70)
    print(f"PIPELINE OZETI -- {RUN_NAME}")
    print("=" * 70)
    status = "OK" if training["exit_code"] == 0 else "BASARISIZ"
    print(f"[egitim] {status}  ({training['duration_seconds']:.0f}s)  -> {MODEL_OUTPUT_DIR}")
    for label, info in benchmarks.items():
        status = "OK" if info["exit_code"] == 0 else "BASARISIZ"
        print(f"[{label}] {status}  ({info['duration_seconds']:.0f}s)  -> {info['output_dir']}")


def main() -> int:
    training_exit_code, training_seconds = run_training()
    training_summary = {"exit_code": training_exit_code, "duration_seconds": training_seconds}

    if training_exit_code != 0:
        print("HATA: egitim basarisiz oldu, benchmark'lara gecilmiyor.", file=sys.stderr)
        print_summary(training_summary, {})
        return 1

    benchmark_summaries = run_benchmarks()

    summary_path = Path(f"benchmarks/{RUN_NAME}/pipeline_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"training": training_summary, "benchmarks": benchmark_summaries}, indent=2),
        encoding="utf-8",
    )

    print_summary(training_summary, benchmark_summaries)
    print(f"\nOzet JSON: {summary_path}")

    any_benchmark_failed = any(info["exit_code"] != 0 for info in benchmark_summaries.values())
    return 1 if any_benchmark_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
