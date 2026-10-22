FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src

# Locked runtime dependencies first, then mlbox itself, into the system Python
RUN uv export --frozen --no-dev --no-emit-project -o requirements.txt \
    && uv pip install --system -r requirements.txt \
    && uv pip install --system --no-deps .

WORKDIR /opt/ml/code
ENTRYPOINT ["python", "-m", "mlbox.worker"]
