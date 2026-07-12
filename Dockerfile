FROM python:3.11-slim

# Run as a non-root user for better container security.
RUN useradd --create-home --uid 10001 bench

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

USER bench

# Run a small demo benchmark so the image is self-verifying on launch.
ENTRYPOINT ["python", "-m", "llm_bench.cli", "run", "--backend", "fake", "--runs", "5"]
