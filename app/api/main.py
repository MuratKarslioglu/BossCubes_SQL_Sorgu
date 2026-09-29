"""Local employer evaluation UI and model API."""

from __future__ import annotations

import csv
import io
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.errors import AppError, InvalidInputError
from app.inference.engine import InferenceEngine, outcome_payload
from app.normalization.text import normalize_whitespace
from app.paths import project_path
from app.schemas.query import QueryRequest
from app.services.feedback import EvaluationStore
from training.evaluate import exact_match, semantic_match

logger = logging.getLogger(__name__)
WEB_ROOT = Path(__file__).parents[1] / "web"


class ExtractInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=20000)
    expected: QueryRequest | None = None


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=32, max_length=32)
    correct: bool
    category: (
        Literal[
            "operation", "target", "filters", "group_by", "order_by", "limit", "status", "other"
        ]
        | None
    ) = None
    note: str = Field(default="", max_length=4000)
    expected: QueryRequest | None = None


def compare(expected: QueryRequest | None, data: dict | None) -> dict | None:
    if expected is None:
        return None
    predicted = QueryRequest.model_validate(data) if data else None
    return {
        "exact_match": predicted is not None and exact_match(expected, predicted),
        "semantic_match": predicted is not None and semantic_match(expected, predicted),
        "different_fields": [
            key
            for key, value in expected.model_dump(mode="json").items()
            if data is None or data.get(key) != value
        ],
    }


def create_app(settings: Settings | None = None, engine_factory=InferenceEngine) -> FastAPI:
    settings = settings or get_settings()
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = EvaluationStore(project_path(settings.feedback_db))
        app.state.engine = engine_factory(settings)
        app.state.ready = False
        app.state.startup_error = None
        try:
            await run_in_threadpool(app.state.engine.start)
            app.state.ready = True
        except Exception:
            logger.exception("model baslatilamadi")
            app.state.startup_error = (
                "Model yuklenemedi; model/adapter ayarlarini ve loglari kontrol edin."
            )
        yield

    app = FastAPI(title="Türkçe Sorgu Değerlendirme", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_ROOT / "index.html")

    @app.get("/health")
    def health():
        return JSONResponse(
            {
                "ready": app.state.ready,
                "error": app.state.startup_error,
                "runtime": app.state.engine.info(),
                "evaluation": app.state.store.summary(),
            },
            status_code=200 if app.state.ready else 503,
        )

    @app.post("/extract")
    def extract_query(body: ExtractInput):
        text = normalize_whitespace(body.text)
        if not text or len(text) > settings.max_input_chars:
            raise HTTPException(422, "Girdi boş veya izin verilen uzunluğu aşıyor.")
        if not app.state.ready:
            raise HTTPException(503, app.state.startup_error)
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "Model başka bir sorgu işliyor. Biraz sonra yeniden deneyin.")
        try:
            outcome = app.state.engine.extract(text)
        except InvalidInputError as exc:
            raise HTTPException(422, exc.message) from exc
        except AppError as exc:
            logger.exception("cikarim basarisiz [%s]", exc.error_type.value)
            raise HTTPException(
                500, "Model isteği işleyemedi. Sunucu loglarını kontrol edin."
            ) from exc
        except Exception as exc:
            logger.exception("beklenmeyen cikarim hatasi")
            raise HTTPException(500, "Model isteği işleyemedi.") from exc
        finally:
            lock.release()
        payload = outcome_payload(outcome)
        payload["runtime"] = app.state.engine.info()
        payload["expected"] = body.expected.model_dump(mode="json") if body.expected else None
        payload["comparison"] = compare(body.expected, payload["data"])
        payload["request_id"] = app.state.store.add(text, payload)
        return payload

    @app.post("/feedback")
    def save_feedback(body: FeedbackInput):
        record = app.state.store.get(body.request_id)
        if record is None:
            raise HTTPException(404, "Deneme kaydı bulunamadı.")
        feedback = body.model_dump(mode="json", exclude={"request_id"})
        expected = body.expected or (
            QueryRequest.model_validate(record["expected"]) if record["expected"] else None
        )
        feedback["comparison"] = compare(expected, record["data"])
        app.state.store.feedback(body.request_id, feedback)
        return {"saved": True, "evaluation": app.state.store.summary()}

    @app.get("/export")
    def export(format: Literal["jsonl", "csv"] = "jsonl"):
        def lines():
            if format == "jsonl":
                for record in app.state.store.records():
                    yield json.dumps(record, ensure_ascii=False) + "\n"
            else:
                output = io.StringIO()
                writer = csv.writer(output)
                yield "\ufeff"
                writer.writerow(["id", "tarih", "prompt", "sonuc", "dogru", "hata", "not", "ms"])
                yield output.getvalue()
                for record in app.state.store.records():
                    output.seek(0)
                    output.truncate()
                    feedback = record["feedback"] or {}
                    cells = [
                        record["request_id"],
                        record["created_at"],
                        record["input"],
                        json.dumps(record["data"], ensure_ascii=False),
                        feedback.get("correct", ""),
                        feedback.get("category", ""),
                        feedback.get("note", ""),
                        record["timing"]["total_ms"],
                    ]
                    # Avoid spreadsheet formula execution from exported user text.
                    writer.writerow(
                        [
                            "'" + str(cell) if str(cell).startswith(("=", "+", "-", "@")) else cell
                            for cell in cells
                        ]
                    )
                    yield output.getvalue()

        return StreamingResponse(
            lines(),
            media_type="application/x-ndjson" if format == "jsonl" else "text/csv",
            headers={"Content-Disposition": f'attachment; filename="degerlendirmeler.{format}"'},
        )

    return app


app = create_app()
