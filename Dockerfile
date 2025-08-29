FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

COPY bcp ./bcp

ENV PORT=8000
EXPOSE 8000

CMD ["python", "-m", "bcp.cli"]

