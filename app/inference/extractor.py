"""Deterministik cikarim: model uretimi + JSON izolasyon/dogrulama (spec §15-17).

Uretim ayarlari spec §15'in gerektirdigi gibi tamamen deterministiktir:
`do_sample=False`, tek beam (`num_beams=1`). `temperature`/`top_p` hic
GECIRILMEZ (None gecirmek yerine) -- bu parametreler yalnizca ornekleme
(sampling) acikken anlamlidir ve do_sample=False iken transformers'in
gereksiz uyari basmasini onler; net sonuc spec'in istedigi ile aynidir.

JSON izolasyon/dogrulama mantigi `app.inference.parsing`'e ayrilmistir
(torch'tan bagimsiz, ayri test edilebilir).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.errors import AppError, InvalidInputError
from app.inference.generation import GenerationTiming, generate_prompts
from app.inference.parsing import ExtractionResult, parse_query_json
from app.normalization.text import normalize_whitespace

if TYPE_CHECKING:
    from app.inference.model_loader import LoadedModel
    from training.formatting import PromptFormatter

logger = logging.getLogger(__name__)


def _generate_from_prompt(
    loaded: LoadedModel,
    prompt: str,
    max_new_tokens: int,
    **options,
) -> tuple[str, GenerationTiming]:
    """Verilen tam prompt'tan ham tamamlama metnini uretir."""
    return generate_prompts(loaded, [prompt], max_new_tokens, **options)[0]


def generate_raw(
    loaded: LoadedModel,
    formatter: PromptFormatter,
    text: str,
    max_new_tokens: int,
    **options,
) -> tuple[str, GenerationTiming]:
    """Modelden ham (JSON izole edilmemis) tamamlama metnini uretir."""
    return _generate_from_prompt(
        loaded, formatter.format_inference(text), max_new_tokens, **options
    )


def _build_repair_prompt(original_prompt: str, bad_output: str) -> str:
    """Spec §17: 'kucuk' bir onarim istemi -- buyuk bir onarim promptuna
    guvenilmez. Modelin kendi hatali ciktisini gorup duzeltmesi istenir."""
    return (
        f"{original_prompt}{bad_output}\n\n"
        "Yukaridaki cikti gecerli degildi. Sadece tek, gecerli bir JSON "
        "nesnesi don.\n"
    )


@dataclass
class ExtractionOutcome:
    """`extract()`in tam sonucu: dogrulanmis veri + benchmark icin zamanlama.

    `timing`, tekrar denendiyse (spec §17) her iki uretim cagrisinin
    TOPLAM gecikmesini ve TOPLAM uretilen token sayisini tasir (gercek
    maliyeti yansitmasi icin); `prompt_tokens` ise SON denemenin degeridir
    (onarim istemi orijinalden uzun oldugundan en anlamli deger budur).
    """

    result: ExtractionResult
    timing: GenerationTiming
    retried: bool


def extract(
    text: str,
    loaded: LoadedModel,
    formatter: PromptFormatter,
    max_new_tokens: int = 192,
    allow_retry: bool = True,
    max_input_tokens: int | None = None,
    profile_tokens: bool = False,
) -> ExtractionOutcome:
    """Spec §16'nin tam akisi + spec §17'nin en fazla bir tekrar politikasi.

    Basarisiz cikti asla sessizce onarilmaz (spec §37): ilk deneme
    basarisiz olursa en fazla bir kez, kucuk bir onarim istemiyle tekrar
    denenir; o da basarisiz olursa acik bir hata donulur.

    `text`, modele gonderilmeden once `normalize_whitespace` ile temizlenir
    (spec §18: agresif degil, yalnizca bosluk/gorunmez karakter normalizasyonu)
    -- bu ayni zamanda yalnizca gorunmez/bosluk karakterlerinden olusan
    girdilerin bos girdi olarak dogru sekilde reddedilmesini saglar.
    """
    request_start = time.perf_counter()
    text = normalize_whitespace(text)
    if not text:
        return ExtractionOutcome(
            result=ExtractionResult.fail(InvalidInputError("girdi metni bos olamaz")),
            timing=GenerationTiming(latency_ms=0.0, prompt_tokens=0, generated_tokens=0),
            retried=False,
        )

    raw_text, timing = generate_raw(
        loaded,
        formatter,
        text,
        max_new_tokens,
        max_input_tokens=max_input_tokens,
        profile_tokens=profile_tokens,
    )
    parse_start = time.perf_counter()

    try:
        result = ExtractionResult.ok(parse_query_json(raw_text))
        timing.parse_ms = (time.perf_counter() - parse_start) * 1000
        timing.total_ms = (time.perf_counter() - request_start) * 1000
        return ExtractionOutcome(result=result, timing=timing, retried=False)
    except AppError as first_error:
        timing.parse_ms = (time.perf_counter() - parse_start) * 1000
        timing.total_ms = (time.perf_counter() - request_start) * 1000
        if not allow_retry:
            return ExtractionOutcome(
                result=ExtractionResult.fail(first_error), timing=timing, retried=False
            )

        logger.info(
            "ilk deneme basarisiz [%s], tek seferlik onarim deneniyor",
            first_error.error_type.value,
        )
        repair_prompt = _build_repair_prompt(formatter.format_inference(text), raw_text)
        retry_text, retry_timing = _generate_from_prompt(
            loaded,
            repair_prompt,
            max_new_tokens,
            max_input_tokens=max_input_tokens,
            profile_tokens=profile_tokens,
        )
        total_timing = GenerationTiming(
            latency_ms=timing.latency_ms + retry_timing.latency_ms,
            prompt_tokens=retry_timing.prompt_tokens,
            generated_tokens=timing.generated_tokens + retry_timing.generated_tokens,
            tokenize_ms=timing.tokenize_ms + retry_timing.tokenize_ms,
            decode_ms=timing.decode_ms + retry_timing.decode_ms,
            parse_ms=timing.parse_ms,
            total_ms=timing.total_ms + retry_timing.total_ms,
            first_token_ms=timing.first_token_ms,
        )
        parse_start = time.perf_counter()
        try:
            result = ExtractionResult.ok(parse_query_json(retry_text))
            return ExtractionOutcome(result=result, timing=total_timing, retried=True)
        except AppError as second_error:
            return ExtractionOutcome(
                result=ExtractionResult.fail(second_error), timing=total_timing, retried=True
            )
        finally:
            parse_ms = (time.perf_counter() - parse_start) * 1000
            total_timing.parse_ms += parse_ms
            total_timing.total_ms = (time.perf_counter() - request_start) * 1000
