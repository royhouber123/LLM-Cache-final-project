# Baseline Framework Choice: GPTCache

**Course final project — LLM cache extension**

## What GPTCache is

GPTCache (github.com/zilliztech/GPTCache, MIT license, ~7,000 GitHub stars)
is the most widely used open-source semantic cache for LLM applications. It
sits between an application and an LLM: incoming prompts are embedded into
vectors, compared against previously answered prompts, and if a similar
enough prompt was already answered, the stored answer is returned instead of
calling the model again.

## Main features

- **Modular pipeline**: embedding generators, vector stores (FAISS, Milvus,
  and others), scalar storage (SQLite and others), similarity evaluators and
  an eviction layer are all separate, swappable components.
- **Adapters** for OpenAI and LangChain, so it drops into existing apps.
- **Tests and docs**: unit tests per component and a documented architecture,
  which made it practical for us to modify safely.
- Pure Python, no GPU required — anyone can reproduce our results.

## Default eviction policy

Cache entries are stored in scalar + vector storage, and an in-memory
eviction index (`MemoryCacheEviction`) decides which entries to drop when
the cache is full. It is a thin wrapper around the `cachetools` library and
offers **LRU (default), LFU, FIFO and RR**. When the cache exceeds
`maxsize`, a batch of entries (`clean_size`, default 20%) is evicted and
deleted from storage via a callback.

## Why it fits this project

The eviction layer is exactly the kind of extension point this project asks
for. It is small (about 60 lines), has a clean abstract interface
(`EvictionBase`), and existing unit tests we could keep passing while adding
our policy. Crucially, all four built-in policies share one blind spot: the
eviction index stores only entry IDs (`cache[id] = True`), so eviction
decisions ignore how *expensive* each cached answer was to generate. For an
LLM cache this matters a lot — answers differ by two orders of magnitude in
generated tokens, so evicting a 2,000-token answer costs the user far more
future time and money than evicting a 20-token one. Our extension (a
cost-aware GDSF policy) targets exactly this gap while staying fully
compatible with the existing `EvictionBase` interface.

We also considered KV-cache-level systems (vLLM, LMCache). They operate at
the GPU memory level and are impressive, but they require GPU hardware,
are much harder to modify, and reproducing benchmarks would be difficult
for a grader. GPTCache lets us make a real, measurable contribution that
anyone can rerun on a laptop.
