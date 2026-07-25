# Reproducible environment for the GDSF eviction-policy benchmarks.
#
# Build:  docker build -t gptcache-gdsf .
# Test:   docker run --rm gptcache-gdsf
# Bench:  docker run --rm -v "$PWD/results:/app/benchmarks/results" \
#             gptcache-gdsf python benchmarks/run_experiments.py --seeds 10

FROM python:3.10-slim

WORKDIR /app
COPY . .

# sqlalchemy + faiss-cpu are needed by the data-manager integration tests.
# cachetools is pinned because the LFU baseline's exact numbers depend on
# its implementation details (GDSF and LRU are unaffected).
RUN pip install --no-cache-dir -e . "cachetools==5.5.2" numpy matplotlib pytest sqlalchemy faiss-cpu

WORKDIR /app

# default: run all eviction unit + integration tests (correctness first)
CMD ["python", "-m", "pytest", "tests/unit_tests/eviction/", \
     "tests/unit_tests/manager/test_eviction.py", "-q", \
     "-o", "addopts=", \
     "--ignore=tests/unit_tests/eviction/test_distributed_cache.py"]
