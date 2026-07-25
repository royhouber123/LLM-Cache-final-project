"""Benchmark harness for GPTCache eviction policies.

We replay a workload (a stream of prompt ids with generation costs) against
GPTCache's eviction layer (MemoryCacheEviction) and measure:

  - hit rate: fraction of requests served from cache
  - cost-weighted hit rate: fraction of total generation tokens that the
    cache saved us (this is the metric that maps to real time and money)
  - simulated end-to-end latency: hits pay a small constant lookup time,
    misses pay a generation time that grows with the number of output tokens
  - harness throughput and eviction count

The latency model is: miss_latency = base_ms + ms_per_token * cost.
Defaults (300ms base, 20ms/token) roughly match a hosted LLM API generating
~50 tokens/second. Both knobs are CLI flags, so results can be recomputed
for faster or slower backends.

Usage example:
  python benchmarks/run_bench.py --policies LRU,LFU --workload zipf \
      --maxsizes 500,1000,2000 --out benchmarks/results/baseline
"""

import argparse
import inspect
import json
import os
import time

import numpy as np

from gptcache.manager.eviction.memory_cache import MemoryCacheEviction

from workloads import WORKLOADS, Workload


def run_one(policy: str, maxsize: int, workload: Workload,
            clean_ratio: float, base_ms: float, ms_per_token: float,
            hit_ms: float, eviction_factory=None,
            return_latencies: bool = False) -> dict:
    """Replay one workload against one (policy, maxsize) configuration.

    ``eviction_factory`` lets callers (e.g. the ablation study) supply a
    custom eviction object instead of the policies built into GPTCache.
    """
    in_cache = set()  # ids currently cached (kept in sync by on_evict)

    def on_evict(keys):
        for k in keys:
            in_cache.discard(k)

    clean_size = max(1, int(maxsize * clean_ratio))
    if eviction_factory is not None:
        cache = eviction_factory(maxsize=maxsize, clean_size=clean_size,
                                 on_evict=on_evict)
    else:
        cache = MemoryCacheEviction(
            policy=policy, maxsize=maxsize, clean_size=clean_size,
            on_evict=on_evict
        )
    # Cost-aware policies accept a cost per entry; vanilla ones do not.
    put_takes_cost = "costs" in inspect.signature(cache.put).parameters

    hits = 0
    saved_tokens = 0.0
    total_tokens = 0.0
    evictions_before = 0
    latencies = np.empty(len(workload.requests), dtype=np.float64)

    t0 = time.perf_counter()
    for idx, (pid, cost) in enumerate(workload.requests):
        total_tokens += cost
        if pid in in_cache:
            cache.get(pid)  # touch so the policy sees the access
            hits += 1
            saved_tokens += cost
            latencies[idx] = hit_ms
        else:
            # miss: "generate" the answer, then insert it
            if put_takes_cost:
                cache.put([pid], costs=[cost])
            else:
                cache.put([pid])
            in_cache.add(pid)
            latencies[idx] = base_ms + ms_per_token * cost
    wall = time.perf_counter() - t0

    n = len(workload.requests)
    if return_latencies:
        return {"latencies": latencies, "policy": policy}
    return {
        "policy": policy,
        "maxsize": maxsize,
        "workload": workload.name,
        "n_requests": n,
        "n_unique": workload.n_unique,
        "hit_rate": hits / n,
        "cost_weighted_hit_rate": saved_tokens / total_tokens,
        "saved_tokens": saved_tokens,
        "total_tokens": total_tokens,
        "latency_mean_ms": float(latencies.mean()),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "evictions": n - hits - len(in_cache),
        "harness_wall_s": wall,
        "harness_ops_per_s": n / wall,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policies", default="LRU,LFU,FIFO,RR",
                    help="comma-separated eviction policies")
    ap.add_argument("--workload", default="zipf", choices=sorted(WORKLOADS))
    ap.add_argument("--maxsizes", default="500,1000,2000",
                    help="comma-separated cache sizes (number of entries)")
    ap.add_argument("--requests", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zipf-s", type=float, default=1.1,
                    help="Zipf skew (zipf workload only)")
    ap.add_argument("--clean-ratio", type=float, default=0.2,
                    help="fraction of the cache evicted per cleanup, "
                         "mirrors GPTCache's default clean_size")
    ap.add_argument("--base-ms", type=float, default=300.0,
                    help="fixed latency of one LLM call on a miss")
    ap.add_argument("--ms-per-token", type=float, default=20.0,
                    help="generation latency per output token")
    ap.add_argument("--hit-ms", type=float, default=5.0,
                    help="latency of a cache hit (lookup only)")
    ap.add_argument("--out", default="benchmarks/results/run",
                    help="output path prefix (.csv and .json are appended)")
    args = ap.parse_args()

    kwargs = {"n_requests": args.requests, "seed": args.seed}
    if args.workload in ("zipf", "oasst"):
        kwargs["s"] = args.zipf_s
    workload = WORKLOADS[args.workload](**kwargs)

    rows = []
    for maxsize in [int(x) for x in args.maxsizes.split(",")]:
        for policy in args.policies.split(","):
            row = run_one(policy.strip(), maxsize, workload,
                          args.clean_ratio, args.base_ms,
                          args.ms_per_token, args.hit_ms)
            rows.append(row)
            print(f"{row['policy']:<6} maxsize={row['maxsize']:<6} "
                  f"hit_rate={row['hit_rate']:.3f} "
                  f"cost_weighted={row['cost_weighted_hit_rate']:.3f} "
                  f"p95={row['latency_p95_ms']:.0f}ms "
                  f"mean={row['latency_mean_ms']:.0f}ms")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    header = list(rows[0].keys())
    with open(args.out + ".csv", "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in header) + "\n")
    with open(args.out + ".json", "w") as f:
        json.dump({"config": vars(args), "workload_params": workload.params,
                   "results": rows}, f, indent=2)
    print(f"\nwrote {args.out}.csv and {args.out}.json")


if __name__ == "__main__":
    main()
