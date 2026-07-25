"""Generate the report figures from the experiment results.

Reads benchmarks/results/full_raw.csv (produced by run_experiments.py) and
writes PNG figures to benchmarks/results/figures/.

Usage (from the benchmarks/ directory):
  python make_plots.py --raw results/full_raw.csv --outdir results/figures
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from run_bench import run_one
from workloads import zipf_workload

# One fixed color/marker per policy, used in every figure (validated palette).
STYLE = {
    "LRU":         {"color": "#2a78d6", "marker": "o", "ls": "-"},
    "LFU":         {"color": "#008300", "marker": "s", "ls": "-"},
    "FIFO":        {"color": "#e87ba4", "marker": "^", "ls": "--"},
    "RR":          {"color": "#1baf7a", "marker": "D", "ls": "-"},
    "GDSF":        {"color": "#eb6834", "marker": "*", "ls": "-"},
    "COST_ONLY":   {"color": "#1baf7a", "marker": "D", "ls": "-"},
    "GDSF_NO_AGE": {"color": "#4a3aa7", "marker": "v", "ls": "-"},
}
INK, MUTED, GRID = "#333333", "#666666", "#e3e3e0"
SIZES = (250, 500, 1000, 2000, 4000)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})


def lw(policy):
    return 2.6 if policy == "GDSF" else 1.8


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def agg(rows, experiment, metric, policies, key_field="maxsize",
        workload=None):
    """-> {policy: {key: (mean, std)}} over seeds."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["experiment"] != experiment or r["policy"] not in policies:
            continue
        if workload and r["workload"] != workload:
            continue
        key = float(r[key_field]) if key_field != "workload" else r[key_field]
        acc[r["policy"]][key].append(float(r[metric]))
    out = {}
    for pol, by_key in acc.items():
        out[pol] = {k: (float(np.mean(v)), float(np.std(v, ddof=1)))
                    for k, v in sorted(by_key.items())}
    return out


def line_panel(ax, data, policies, ylabel, legend_loc="lower right"):
    for pol in policies:
        xs = list(data[pol].keys())
        means = np.array([data[pol][x][0] for x in xs])
        stds = np.array([data[pol][x][1] for x in xs])
        st = STYLE[pol]
        ax.plot(xs, means, color=st["color"], marker=st["marker"],
                linestyle=st["ls"],
                linewidth=lw(pol), markersize=7 if pol != "GDSF" else 11,
                label=pol)
        ax.fill_between(xs, means - stds, means + stds,
                        color=st["color"], alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks(SIZES)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("cache size (entries)")
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/full_raw.csv")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows = load_rows(args.raw)
    main_pols = ["FIFO", "RR", "LRU", "LFU", "GDSF"]

    # Fig 1: cost-weighted hit rate vs cache size
    fig, ax = plt.subplots(figsize=(7, 4.4))
    data = agg(rows, "E1_size_sweep", "cost_weighted_hit_rate", main_pols)
    line_panel(ax, data, main_pols, "cost-weighted hit rate\n(fraction of tokens saved)")
    ax.set_title("Tokens saved by the cache, per eviction policy")
    fig.savefig(os.path.join(args.outdir, "fig1_cost_weighted_vs_size.png"))
    plt.close(fig)

    # Fig 2: raw hit rate vs cache size
    fig, ax = plt.subplots(figsize=(7, 4.4))
    data = agg(rows, "E1_size_sweep", "hit_rate", main_pols)
    line_panel(ax, data, main_pols, "hit rate")
    ax.set_title("Raw hit rate, per eviction policy")
    fig.savefig(os.path.join(args.outdir, "fig2_hit_rate_vs_size.png"))
    plt.close(fig)

    # Fig 3: mean and p95 latency vs cache size (two panels, one y-axis each)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, label in zip(
            axes, ("latency_mean_ms", "latency_p95_ms"),
            ("mean latency (ms)", "p95 latency (ms)")):
        data = agg(rows, "E1_size_sweep", metric, main_pols)
        line_panel(ax, data, main_pols, label, legend_loc="upper right")
    axes[0].set_title("Mean request latency")
    axes[1].set_title("p95 request latency")
    fig.savefig(os.path.join(args.outdir, "fig3_latency_vs_size.png"))
    plt.close(fig)

    # Fig 4: skew sweep
    fig, ax = plt.subplots(figsize=(7, 4.4))
    pols = ["LRU", "LFU", "GDSF"]
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["experiment"] == "E2_skew_sweep" and r["policy"] in pols:
            s = float(r["workload"].split("_s")[1])
            acc[r["policy"]][s].append(float(r["cost_weighted_hit_rate"]))
    for pol in pols:
        xs = sorted(acc[pol])
        means = np.array([np.mean(acc[pol][x]) for x in xs])
        stds = np.array([np.std(acc[pol][x], ddof=1) for x in xs])
        st = STYLE[pol]
        ax.plot(xs, means, color=st["color"], marker=st["marker"],
                linewidth=lw(pol), markersize=7 if pol != "GDSF" else 11,
                label=pol)
        ax.fill_between(xs, means - stds, means + stds, color=st["color"],
                        alpha=0.15, linewidth=0)
    ax.set_xlabel("Zipf skew parameter s (higher = more repetitive traffic)")
    ax.set_ylabel("cost-weighted hit rate")
    ax.set_title("Sensitivity to workload skew (cache size 1000)")
    ax.legend(loc="lower right", fontsize=10)
    fig.savefig(os.path.join(args.outdir, "fig4_skew_sweep.png"))
    plt.close(fig)

    # Fig 5: ablation bars, stationary vs drifting popularity (maxsize 1000)
    abl_pols = ["LRU", "LFU", "COST_ONLY", "GDSF_NO_AGE", "GDSF"]
    abl_labels = ["LRU", "LFU\n(freq only)", "cost only",
                  "freq x cost\n(no aging)", "GDSF\n(full)"]

    def ablation_panel(ax, workload_name, title, ymin=None):
        means, stds = [], []
        for pol in abl_pols:
            vals = [float(r["cost_weighted_hit_rate"]) for r in rows
                    if r["experiment"] == "E3_ablation"
                    and r["policy"] == pol and int(r["maxsize"]) == 1000
                    and r["workload"] == workload_name]
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1))
        xs = np.arange(len(abl_pols))
        colors = [STYLE[p]["color"] for p in abl_pols]
        ax.bar(xs, means, yerr=stds, width=0.62, color=colors,
               error_kw={"ecolor": INK, "capsize": 3})
        for x, m, s in zip(xs, means, stds):
            ax.annotate(f"{m:.3f}", (x, m + s), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(abl_labels, fontsize=9.5)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    ablation_panel(axes[0], "zipf_s1.1", "stationary popularity")
    ablation_panel(axes[1], "zipf_s1.1_drift", "popularity drifts mid-stream")
    axes[0].set_ylabel("cost-weighted hit rate")
    fig.suptitle("Ablation: contribution of each GDSF ingredient "
                 "(cache size 1000)", y=1.02)
    fig.savefig(os.path.join(args.outdir, "fig5_ablation.png"))
    plt.close(fig)

    # Fig 6: latency CDF (recomputed live: seed 1, maxsize 1000)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    wl = zipf_workload(n_requests=50_000, s=1.1, seed=1)
    for pol in ("LRU", "LFU", "GDSF"):
        lat = np.sort(run_one(pol, 1000, wl, 0.2, 300.0, 20.0, 5.0,
                              return_latencies=True)["latencies"])
        cdf = np.arange(1, len(lat) + 1) / len(lat)
        st = STYLE[pol]
        ax.plot(lat, cdf, color=st["color"], linewidth=lw(pol), label=pol)
    ax.set_xscale("log")
    ax.set_ylim(0.75, 1.005)
    ax.set_xlabel("request latency (ms, log scale)")
    ax.set_ylabel("fraction of requests")
    ax.set_title("Latency distribution, tail region\n"
                 "(zipf, cache size 1000, seed 1; hits are the flat part left of 1s)")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(args.outdir, "fig6_latency_cdf.png"))
    plt.close(fig)

    # Fig 7: harness throughput on the uncacheable workload (overhead)
    fig, ax = plt.subplots(figsize=(6, 4))
    pols = ["LRU", "LFU", "GDSF"]
    means, stds = [], []
    for pol in pols:
        vals = [float(r["harness_ops_per_s"]) for r in rows
                if r["experiment"] == "E4_overhead" and r["policy"] == pol]
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1))
    xs = np.arange(len(pols))
    ax.bar(xs, means, yerr=stds, width=0.55,
           color=[STYLE[p]["color"] for p in pols],
           error_kw={"ecolor": INK, "capsize": 3})
    for x, m in zip(xs, means):
        ax.annotate(f"{m/1000:.0f}k", (x, m), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(pols)
    ax.set_ylabel("cache operations / second")
    ax.set_title("Cache-layer throughput, 0%-hit workload\n(pure overhead, higher is better)")
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(args.outdir, "fig7_overhead.png"))
    plt.close(fig)

    print(f"wrote 7 figures to {args.outdir}/")


if __name__ == "__main__":
    main()
