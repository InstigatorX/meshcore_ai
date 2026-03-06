FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# (Optional but nice) install minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mc_ai.py .

# Run as non-root by default (safer).
# If you need serial, ensure device permissions on host or run as root.
RUN useradd -m -u 10001 appuser
USER appuser

ENTRYPOINT ["python", "/app/mc_ai.py"]
