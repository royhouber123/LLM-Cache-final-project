"""Workload generators for the cache benchmarks.

A workload is a list of requests. Each request is a tuple:
    (prompt_id, cost)
where cost is the number of output tokens the LLM would need to generate
if the prompt is a cache miss. We use tokens as the "cost" unit because
generation time and API price both grow with the number of output tokens.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Workload:
    name: str
    requests: List[Tuple[int, float]]  # (prompt_id, cost_in_tokens)
    n_unique: int
    params: dict


def _make_costs(rng: np.random.Generator, n_unique: int, kind: str) -> np.ndarray:
    """Assign a fixed generation cost (output tokens) to every unique prompt.

    kind:
      "lognormal" - realistic skew: most answers are short, a few are very long
      "short"     - all answers short (10-50 tokens)
      "long"      - all answers long (1000-3000 tokens)
    """
    if kind == "lognormal":
        costs = rng.lognormal(mean=5.5, sigma=1.0, size=n_unique)
        return np.clip(costs, 10, 4000)
    if kind == "short":
        return rng.uniform(10, 50, size=n_unique)
    if kind == "long":
        return rng.uniform(1000, 3000, size=n_unique)
    raise ValueError(f"unknown cost kind: {kind}")


def zipf_workload(
    n_requests: int = 50_000,
    n_unique: int = 5_000,
    s: float = 1.1,
    cost_kind: str = "lognormal",
    seed: int = 42,
    drift: bool = False,
) -> Workload:
    """Zipf-distributed prompt popularity (a few prompts are asked a lot,
    a long tail is asked rarely). This is the standard model for cache
    workloads. Costs are drawn independently of popularity, so popular
    prompts can be either cheap or expensive.

    With ``drift=True`` the popularity ranking is re-shuffled halfway
    through the stream: prompts that were hot in the first half become
    unpopular and vice versa. Real traffic drifts like this (topics come
    and go), and it is the situation aging mechanisms are designed for.
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_unique + 1)
    probs = 1.0 / ranks**s
    probs /= probs.sum()
    ids = rng.choice(n_unique, size=n_requests, p=probs)
    if drift:
        remap = rng.permutation(n_unique)
        half = n_requests // 2
        ids[half:] = remap[ids[half:]]
    costs = _make_costs(rng, n_unique, cost_kind)
    return Workload(
        name=f"zipf_s{s}" + ("_drift" if drift else ""),
        requests=[(int(i), float(costs[i])) for i in ids],
        n_unique=n_unique,
        params={"n_requests": n_requests, "n_unique": n_unique, "s": s,
                "cost_kind": cost_kind, "seed": seed, "drift": drift},
    )


def repetitive_short_workload(
    n_requests: int = 50_000, n_unique: int = 50, seed: int = 42
) -> Workload:
    """A small set of short prompts repeated over and over.
    Stresses the hit/miss bookkeeping: almost everything should be a hit.
    """
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, n_unique, size=n_requests)
    costs = _make_costs(rng, n_unique, "short")
    return Workload(
        name="repetitive_short",
        requests=[(int(i), float(costs[i])) for i in ids],
        n_unique=n_unique,
        params={"n_requests": n_requests, "n_unique": n_unique, "seed": seed},
    )


def novel_long_workload(n_requests: int = 20_000, seed: int = 42) -> Workload:
    """Every prompt is new and the answers are long.
    Hit rate is 0% by construction, so this measures pure cache overhead
    (how much time the cache adds when it never helps).
    """
    rng = np.random.default_rng(seed)
    costs = _make_costs(rng, n_requests, "long")
    return Workload(
        name="novel_long",
        requests=[(i, float(costs[i])) for i in range(n_requests)],
        n_unique=n_requests,
        params={"n_requests": n_requests, "seed": seed},
    )


WORKLOADS = {
    "zipf": zipf_workload,
    "repetitive_short": repetitive_short_workload,
    "novel_long": novel_long_workload,
}
