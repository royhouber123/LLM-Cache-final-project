"""Full experiment suite for the final report.

Runs every experiment over multiple random seeds so we can report means,
standard deviations and bootstrap confidence intervals instead of single
lucky numbers. Writes one CSV with every raw run plus a JSON summary with
the aggregated statistics.

Experiments:
  E1 main    : zipf workload, cache-size sweep, all policies vs GDSF
  E2 skew    : zipf skew (s) sweep at a fixed cache size
  E3 ablation: LFU / COST_ONLY / GDSF_NO_AGE / GDSF at fixed sizes
  E4 overhead: novel_long workload (0% hits) - cache overhead & throughput
  E5 sanity  : repetitive_short workload - every policy must be ~100% hits
  E6 real    : oasst workload (real OASST1 response lengths as costs),
               cache-size sweep, same comparison as E1

Usage:
  python benchmarks/run_experiments.py --seeds 10 --out benchmarks/results/full
"""

import argparse
import json
import os

import numpy as np

from ablation import ABLATION_POLICIES
from run_bench import run_one
from workloads import (novel_long_workload, oasst_workload,
                       repetitive_short_workload, zipf_workload)

LATENCY = {"base_ms": 300.0, "ms_per_token": 20.0, "hit_ms": 5.0}
CLEAN_RATIO = 0.2


def run_config(policy, maxsize, workload, factory=None):
    return run_one(policy, maxsize, workload, CLEAN_RATIO,
                   LATENCY["base_ms"], LATENCY["ms_per_token"],
                   LATENCY["hit_ms"], eviction_factory=factory)


def bootstrap_ci(diffs, n_boot=10_000, seed=0):
    """95% bootstrap confidence interval for the mean of paired differences."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    means = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--requests", type=int, default=50_000)
    ap.add_argument("--out", default="benchmarks/results/full")
    args = ap.parse_args()

    seeds = list(range(1, args.seeds + 1))
    rows = []

    def record(experiment, seed, row):
        row = dict(row)
        row["experiment"] = experiment
        row["seed"] = seed
        rows.append(row)

    # ---- E1: main comparison, cache-size sweep -------------------------
    print("E1: cache-size sweep (zipf s=1.1)")
    for seed in seeds:
        wl = zipf_workload(n_requests=args.requests, s=1.1, seed=seed)
        for maxsize in (250, 500, 1000, 2000, 4000):
            for policy in ("LRU", "LFU", "FIFO", "RR", "GDSF"):
                record("E1_size_sweep", seed, run_config(policy, maxsize, wl))
        print(f"  seed {seed} done")

    # ---- E2: skew sweep -------------------------------------------------
    print("E2: zipf skew sweep (maxsize=1000)")
    for seed in seeds:
        for s in (0.8, 1.0, 1.2, 1.4):
            wl = zipf_workload(n_requests=args.requests, s=s, seed=seed)
            for policy in ("LRU", "LFU", "GDSF"):
                record("E2_skew_sweep", seed, run_config(policy, 1000, wl))
        print(f"  seed {seed} done")

    # ---- E3: ablation (stationary and drifting popularity) ---------------
    print("E3: ablation (maxsize 500 and 1000, with and without drift)")
    for seed in seeds:
        for drift in (False, True):
            wl = zipf_workload(n_requests=args.requests, s=1.1, seed=seed,
                               drift=drift)
            for maxsize in (500, 1000):
                for policy, factory in ABLATION_POLICIES.items():
                    record("E3_ablation", seed,
                           run_config(policy, maxsize, wl, factory))
        print(f"  seed {seed} done")

    # ---- E4: overhead on an uncacheable workload ------------------------
    print("E4: overhead (novel_long, 0% hit rate possible)")
    for seed in seeds[:5]:
        wl = novel_long_workload(n_requests=20_000, seed=seed)
        for policy in ("LRU", "LFU", "GDSF"):
            record("E4_overhead", seed, run_config(policy, 1000, wl))

    # ---- E5: sanity check ------------------------------------------------
    print("E5: sanity (repetitive_short, ~100% hits expected)")
    for seed in seeds[:3]:
        wl = repetitive_short_workload(n_requests=args.requests, seed=seed)
        for policy in ("LRU", "LFU", "FIFO", "RR", "GDSF"):
            record("E5_sanity", seed, run_config(policy, 1000, wl))

    # ---- E6: real response lengths (OASST1) ------------------------------
    print("E6: real-cost sweep (oasst, s=1.1)")
    for seed in seeds:
        wl = oasst_workload(n_requests=args.requests, s=1.1, seed=seed)
        for maxsize in (250, 500, 1000, 2000):
            for policy in ("LRU", "LFU", "FIFO", "RR", "GDSF"):
                record("E6_real_costs", seed, run_config(policy, maxsize, wl))
        print(f"  seed {seed} done")

    # ---- write raw CSV ---------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    header = list(rows[0].keys())
    with open(args.out + "_raw.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in header) + "\n")

    # ---- aggregate + significance ----------------------------------------
    summary = {}
    metrics = ("hit_rate", "cost_weighted_hit_rate", "latency_mean_ms",
               "latency_p95_ms", "latency_p99_ms", "harness_ops_per_s")

    def group(experiment):
        out = {}
        for row in rows:
            if row["experiment"] != experiment:
                continue
            key = (row["workload"], row["maxsize"], row["policy"])
            out.setdefault(key, []).append(row)
        return out

    for experiment in ("E1_size_sweep", "E2_skew_sweep", "E3_ablation",
                       "E4_overhead", "E5_sanity", "E6_real_costs"):
        agg = {}
        for (wl_name, maxsize, policy), rs in group(experiment).items():
            stats = {}
            for m in metrics:
                vals = np.array([r[m] for r in rs], dtype=float)
                stats[m] = {"mean": float(vals.mean()),
                            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                            "n": len(vals)}
            agg[f"{wl_name}|{maxsize}|{policy}"] = stats
        summary[experiment] = agg

    # paired GDSF-vs-baseline differences per seed
    def significance(experiment):
        sig = {}
        grouped = group(experiment)
        for baseline in ("LRU", "LFU"):
            for metric, better in (("cost_weighted_hit_rate", "higher"),
                                   ("latency_mean_ms", "lower"),
                                   ("latency_p95_ms", "lower")):
                for maxsize in (500, 1000, 2000):
                    key_g = [k for k in grouped if k[1] == maxsize and k[2] == "GDSF"]
                    key_b = [k for k in grouped if k[1] == maxsize and k[2] == baseline]
                    if not key_g or not key_b:
                        continue
                    g = {r["seed"]: r[metric] for r in grouped[key_g[0]]}
                    b = {r["seed"]: r[metric] for r in grouped[key_b[0]]}
                    diffs = [g[s] - b[s] for s in sorted(g) if s in b]
                    lo, hi = bootstrap_ci(diffs)
                    rel = float(np.mean(diffs) / np.mean(list(b.values())))
                    sig[f"GDSF_vs_{baseline}|{metric}|maxsize={maxsize}"] = {
                        "mean_diff": float(np.mean(diffs)),
                        "relative_change": rel,
                        "ci95": [lo, hi],
                        "better_direction": better,
                        "significant": (lo > 0 if better == "higher" else hi < 0),
                        "n_seeds": len(diffs),
                    }
        return sig

    sig = significance("E1_size_sweep")
    summary["significance_E1"] = sig
    summary["significance_E6"] = significance("E6_real_costs")

    with open(args.out + "_summary.json", "w") as f:
        json.dump({"config": vars(args), "latency_model": LATENCY,
                   "clean_ratio": CLEAN_RATIO, "summary": summary}, f, indent=2)

    print(f"\nwrote {args.out}_raw.csv and {args.out}_summary.json")
    for exp, exp_sig in (("E1", sig), ("E6", summary["significance_E6"])):
        print(f"\n--- GDSF vs baselines ({exp}, paired over seeds, 95% bootstrap CI) ---")
        for k, v in exp_sig.items():
            mark = "SIGNIFICANT" if v["significant"] else "not significant"
            print(f"{k:<55} rel={v['relative_change']:+.1%}  "
                  f"CI=[{v['ci95'][0]:.4g}, {v['ci95'][1]:.4g}]  {mark}")


if __name__ == "__main__":
    main()
