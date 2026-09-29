#!/usr/bin/env python3
"""Export merged weights for PyTorch or later GGUF/TensorRT conversion."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter_path", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output_dir bos olmali; mevcut modelin uzerine yazilmaz")
    from app.errors import AppError
    from app.inference.model_loader import load_adapter, load_base_model, prepare_for_inference

    try:
        loaded = load_base_model(args.model_name, device="cpu", dtype=args.dtype)
        loaded.model = load_adapter(loaded.model, args.adapter_path)
        prepare_for_inference(loaded, merge_adapter=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        loaded.model.save_pretrained(args.output_dir, safe_serialization=True)
        loaded.tokenizer.save_pretrained(args.output_dir)
        manifest = {
            "base_model": args.model_name,
            "adapter": str(args.adapter_path),
            "dtype": args.dtype,
            "created_at": datetime.now(UTC).isoformat(),
        }
        (args.output_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2))
    except AppError as exc:
        print(f"HATA: {exc.message}", file=sys.stderr)
        return 1
    print(f"Birlesik model: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
