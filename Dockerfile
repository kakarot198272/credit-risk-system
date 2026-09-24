FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-linux.lock .
RUN pip install --no-cache-dir -r requirements-linux.lock
RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser . .
RUN mkdir -p /app/artifacts/demo /app/logs && chown -R appuser:appuser /app/artifacts /app/logs /app/models
USER appuser
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
