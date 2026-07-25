"""Ablation variants of the GDSF policy.

GDSF combines three ingredients: frequency, cost, and aging (the inflation
value L). To understand how much each ingredient contributes, we build
stripped-down variants and benchmark them under the same workloads:

  LFU           frequency only                    (built into GPTCache)
  COST_ONLY     cost only:        H = L + cost
  GDSF_NO_AGE   frequency * cost, no aging:  H = freq * cost
  GDSF          the full policy:  H = L + freq * cost

These variants are benchmark-only on purpose: we did not want to clutter
the library with policies that exist just for the ablation study.
"""

import heapq

from gptcache.manager.eviction.gdsf import GDSFCache
from gptcache.manager.eviction.memory_cache import (
    MemoryCacheEviction,
    popitem_wrapper,
)


class CostOnlyCache(GDSFCache):
    """H = L + cost / size (frequency ignored)."""

    def _push(self, key, value):
        size = self.getsizeof(value) or 1
        prio = self._inflation + self._cost_of(value) / size
        self._prio[key] = prio
        heapq.heappush(self._heap, (prio, next(self._counter), key))


class NoAgingGDSFCache(GDSFCache):
    """H = freq * cost / size (no inflation term)."""

    def _push(self, key, value):
        size = self.getsizeof(value) or 1
        prio = self._freq[key] * self._cost_of(value) / size
        self._prio[key] = prio
        heapq.heappush(self._heap, (prio, next(self._counter), key))

    def popitem(self):
        key, value = super().popitem()
        self._inflation = 0.0  # aging disabled
        return key, value


class VariantEviction(MemoryCacheEviction):
    """MemoryCacheEviction wrapper around an arbitrary cache class, so the
    ablation variants go through the exact same code path as the real
    policies (including the popitem/on_evict wrapping)."""

    # pylint: disable=super-init-not-called
    def __init__(self, cache_cls, policy_name, maxsize, clean_size, on_evict):
        self._policy = policy_name
        self._cache = cache_cls(maxsize=maxsize)
        self._cache.popitem = popitem_wrapper(
            self._cache.popitem, on_evict, clean_size
        )


def make_factory(cache_cls, policy_name):
    def factory(maxsize, clean_size, on_evict):
        return VariantEviction(cache_cls, policy_name, maxsize, clean_size,
                               on_evict)
    return factory


ABLATION_POLICIES = {
    # name -> eviction_factory (None means: use the built-in policy)
    "LRU": None,
    "LFU": None,
    "COST_ONLY": make_factory(CostOnlyCache, "COST_ONLY"),
    "GDSF_NO_AGE": make_factory(NoAgingGDSFCache, "GDSF_NO_AGE"),
    "GDSF": None,
}
