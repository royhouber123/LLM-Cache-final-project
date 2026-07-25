# How to benchmark

This folder contains the performance test suite we wrote for our final
project. It compares GPTCache's built-in eviction policies (LRU, LFU, FIFO,
RR) against our new cost-aware policy (GDSF).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e . numpy matplotlib pytest
```

(`matplotlib` is only needed for `make_plots.py`.)

## Running a benchmark

From the repository root:

```bash
python benchmarks/run_bench.py --policies LRU,LFU,GDSF --workload zipf \
    --maxsizes 500,1000,2000 --out benchmarks/results/my_run
```

This prints a summary table and writes `my_run.csv` and `my_run.json` with
the full numbers and the exact configuration, so every run can be reproduced.

## Workloads

We use three workload profiles (see `workloads.py`):

- **zipf** — prompt popularity follows a Zipf distribution (a few prompts are
  asked very often, most are asked rarely). This is the standard model for
  cache workloads. Each unique prompt gets a generation cost (output tokens)
  drawn from a log-normal distribution, so most answers are short and a few
  are very long, like in real LLM traffic.
- **repetitive_short** — 50 short prompts repeated 50,000 times. Almost every
  request should be a hit. Used as a sanity check for the hit/miss logic.
- **novel_long** — every prompt is new, so the hit rate is 0% by
  construction. This measures the overhead the cache adds when it never
  helps.
- **oasst** — same Zipf popularity, but the generation costs are real: for
  each of 3,634 unique first-turn English prompts from the OASST1 dataset
  (OpenAssistant/oasst1, Apache-2.0) we use the length of the actual
  assistant reply (chars / 4 ≈ tokens) as its cost. The derived cost file
  is committed at `data/oasst1_costs.csv`, so no download is needed. This
  replaces the log-normal cost assumption with an empirical response-length
  distribution. (OASST1 prompts are nearly all unique, so the popularity
  pattern still has to be synthetic.)

All workloads are generated with a fixed random seed (default 42, can be
changed with `--seed`), so runs are fully deterministic.

## Metrics

- **hit_rate** — fraction of requests answered from the cache.
- **cost_weighted_hit_rate** — fraction of the total generation tokens that
  the cache saved. This is the metric we care about most: saving one
  2,000-token answer is worth as much as saving two hundred 10-token
  answers, in both time and money.
- **latency (mean / p50 / p95 / p99)** — simulated end-to-end latency. A hit
  costs a small constant lookup time (`--hit-ms`, default 5 ms). A miss
  costs `--base-ms + --ms-per-token * tokens` (defaults: 300 ms + 20
  ms/token, roughly a hosted API generating ~50 tokens/second). Because the
  model is a simple linear function, the results can be recomputed for any
  faster or slower backend.
- **evictions, harness_ops_per_s, harness_wall_s** — bookkeeping and
  overhead of the cache layer itself.

## Why simulated latency?

We replay workloads directly against GPTCache's eviction layer
(`MemoryCacheEviction`) with a mocked LLM instead of calling a real model.
This keeps runs free, fast (~50k requests in seconds), deterministic, and
runnable on any laptop, which we consider more important for reproducibility
than absolute latency numbers. The relative comparison between policies is
unaffected: all policies see the exact same request stream and the same
latency model.

## Running the unit tests

```bash
python -m pytest tests/unit_tests/eviction/ -q -o addopts="" \
    --ignore=tests/unit_tests/eviction/test_distributed_cache.py
```

(The ignored file needs a running Redis server and is unrelated to eviction
policies in memory.)
