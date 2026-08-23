FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

COPY proxy.py ./

EXPOSE 8989

# BOB_API_KEY must be provided at runtime via -e or .env
CMD ["uv", "run", "uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "8989", "--reload"]
