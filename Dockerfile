# CPU image only; Jetson requires a JetPack-compatible NVIDIA runtime.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /workspace
COPY requirements ./requirements
RUN pip install -r requirements/web.txt -c requirements/constraints.txt
COPY pyproject.toml main.py ./
COPY app ./app
COPY training ./training
COPY scripts ./scripts
COPY configs ./configs
COPY model_adapter ./model_adapter
EXPOSE 8000

FROM base AS test
RUN pip install pytest==9.1.1 httpx==0.28.1 tqdm
COPY tests ./tests
CMD ["pytest"]

FROM base AS api
RUN pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements/inference.txt -c requirements/constraints.txt
CMD ["python", "main.py", "serve", "--host", "0.0.0.0"]
