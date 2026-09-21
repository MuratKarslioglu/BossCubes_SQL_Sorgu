#!/usr/bin/env python3
"""Kaggle 'turkish-query-sql' setini proje semamiza (OOD eval) donusturur.

Kaynak: `datasets/_kaggle_transfer/turkish_query_sql/turkish_query_answ_sql.csv`
(Input, Query kolonlari -- Turkce dogal dil sorusu + elle yazilmis ham SQL).
Bu, sentetik uretecimizden GELMEYEN tek gercek-dunya Turkce kaynagimiz --
bu yuzden TUR2SQL/TURSpider ile ayni ilke uygulanir: donusturulen satirlar
`datasets/ood_eval/` altina yazilir, ASLA egitime eklenmez (yalnizca
`scripts/benchmark.py --test_file` ile okunur). Sebep: SQL'ler elle
yazilmis, tutarsiz (bazi satirlarda tablo adi hatali -- ör. "Personel"
tanimsizken FROM/JOIN "employees" kullaniliyor, akilli tirnak ’ ” bazen
duz tirnak yerine gecmis) -- yani hem etiket kalitesi hem de coverage
egitim icin guvenilir degil, ama modelin gercek kullanicilarin nasil soru
sordugunu olcmek icin degerli.

Donusum, `import_external_sql_dataset.py`'deki TURSpider mantigini
(sqlglot AST -> semamiz, tek tablo/basit filtre/agregasyon) aynen
kullanir -- kaynak farkli (Spider TSV yerine bu CSV) ama hedef SQL
sekli (ham, cok-domain, join'ler yaygin) ayni, mantigi tekrar yazmaya
gerek yok.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from import_external_sql_dataset import (  # noqa: E402
    ConversionStats,
    _validate,
    convert_turspider_row,
    write_jsonl,
)


def process_kaggle_csv(csv_path: Path, stats: ConversionStats) -> list[dict]:
    results = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = (row.get("Input") or "").strip()
            sql = (row.get("Query") or "").strip()
            if not question or not sql:
                continue
            stats.read += 1
            converted = convert_turspider_row(sql)
            if converted is None:
                stats.incompatible += 1
                continue
            validated = _validate(converted, stats)
            if validated is not None:
                results.append({"input": question, "output": validated})
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("datasets/_kaggle_transfer/turkish_query_sql/turkish_query_answ_sql.csv"),
    )
    parser.add_argument("--ood-out-dir", type=Path, default=Path("datasets/ood_eval"))
    args = parser.parse_args()

    stats = ConversionStats()
    rows = process_kaggle_csv(args.csv, stats)
    out_path = args.ood_out_dir / "kaggle_turkish_real_nl_ood.jsonl"
    write_jsonl(rows, out_path, "kaggle_turkish_query_sql", None)
    print(stats.report("Kaggle turkish-query-sql (OOD)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
