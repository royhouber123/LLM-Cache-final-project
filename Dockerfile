# Reproducible environment for the GDSF eviction-policy benchmarks.
#
# Build:  docker build -t gptcache-gdsf .
# Test:   docker run --rm gptcache-gdsf
# Bench:  docker run --rm -v "$PWD/results:/app/benchmarks/results" \
#             gptcache-gdsf python benchmarks/run_experiments.py --seeds 10

FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e . numpy matplotlib pytest

WORKDIR /app

# default: run the eviction unit tests (correctness first)
CMD ["python", "-m", "pytest", "tests/unit_tests/eviction/", "-q", \
     "-o", "addopts=", \
     "--ignore=tests/unit_tests/eviction/test_distributed_cache.py"]
