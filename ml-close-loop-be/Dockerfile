FROM python:3.12-slim

WORKDIR /app

# Symlink so host-created venvs (which point to /usr/bin/python3) resolve inside the container.
RUN ln -s /usr/local/bin/python3 /usr/bin/python3 && \
    ln -s /usr/local/bin/python3.12 /usr/bin/python3.12

COPY pyproject.toml .
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini .
COPY docker-entrypoint.sh docker-worker-entrypoint.sh ./

RUN chmod +x docker-entrypoint.sh docker-worker-entrypoint.sh \
    && pip install --no-cache-dir ".[intake]"

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["./docker-entrypoint.sh"]
