FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy everything needed to install and run
COPY pyproject.toml ./
COPY src ./src
COPY models ./models

# Install runtime deps (and your package)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "pm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
