FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY docker-entrypoint.sh docker-worker-entrypoint.sh ./

RUN chmod +x docker-entrypoint.sh docker-worker-entrypoint.sh \
    && pip install --no-cache-dir "."

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["./docker-entrypoint.sh"]
