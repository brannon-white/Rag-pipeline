# Multi-stage build: dependencies are synced in a builder stage (cached
# separately from source changes) and only the resulting venv + app code
# cross into the slim runtime stage -- uv, pip caches, and build tooling
# never do. Deploy.yml tags every image with the git SHA, never `latest`, so
# App Runner always deploys something reproducible and rollback-able.

FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies first, without the project itself, so this layer only
# invalidates when pyproject.toml/uv.lock change -- not on every source edit.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Now the project itself, non-editable: the runtime image should not depend
# on these exact source paths existing at their build-time locations.
# README.md is required here too -- hatchling's build backend reads
# pyproject.toml's `readme` field and fails the wheel build without it.
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM python:3.13-slim AS runtime

# Never run as root -- this is the app's only network-facing surface.
RUN groupadd --system appuser && useradd --system --gid appuser --no-create-home appuser

WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/migrations ./migrations

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

# /healthz is shallow (no DB) by design -- see db/pool.py's docstring -- so
# it's also what App Runner's own health check config points at, not just
# this. No HEALTHCHECK instruction here: App Runner does its own polling and
# a second, differently-configured check would just be redundant noise.
CMD ["uvicorn", "trialrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
