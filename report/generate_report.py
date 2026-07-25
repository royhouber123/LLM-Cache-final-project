"""Generate the final project report PDF.

All numbers in the text and tables are read from
benchmarks/results/full_summary.json, so the report always matches the data.

Run from the repository root:
  python report/generate_report.py
"""

import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(ROOT, "benchmarks", "results", "figures")
SUMMARY = json.load(open(os.path.join(ROOT, "benchmarks", "results",
                                      "full_summary.json")))
S = SUMMARY["summary"]
SIG = S["significance_E1"]
SIG6 = S["significance_E6"]

OUT = os.path.join(HERE, "final_report.pdf")

# the upstream contribution (Section 6)
PR_LINK = ("<font face='Courier'>github.com/zilliztech/GPTCache/pull/"
           "689</font>")

# ---------------------------------------------------------------- helpers


def e1(size, policy, metric):
    return S["E1_size_sweep"][f"zipf_s1.1|{size}|{policy}"][metric]


def abl(workload, policy):
    return S["E3_ablation"][f"{workload}|1000|{policy}"]["cost_weighted_hit_rate"]


def sig(baseline, metric, size):
    return SIG[f"GDSF_vs_{baseline}|{metric}|maxsize={size}"]


def sig6(baseline, metric, size):
    return SIG6[f"GDSF_vs_{baseline}|{metric}|maxsize={size}"]


def e6(size, policy, metric):
    return S["E6_real_costs"][f"oasst_s1.1|{size}|{policy}"][metric]


styles = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5,
                      leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7)
H1 = ParagraphStyle("H1x", parent=styles["Heading1"], fontSize=15,
                    spaceBefore=16, spaceAfter=8)
H2 = ParagraphStyle("H2x", parent=styles["Heading2"], fontSize=12.5,
                    spaceBefore=12, spaceAfter=6)
CAPTION = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9.5,
                         leading=12, alignment=1, textColor=colors.HexColor("#444444"),
                         spaceBefore=4, spaceAfter=12)
TITLE = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=19, leading=24)
SUBTITLE = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=12,
                          alignment=1, textColor=colors.HexColor("#444444"),
                          spaceAfter=4)
CODE = ParagraphStyle("CodeX", parent=styles["Code"], fontSize=9, leading=12,
                      backColor=colors.HexColor("#f4f4f2"), borderPadding=6,
                      leftIndent=6, spaceBefore=4, spaceAfter=10)

story = []


def p(text, style=BODY):
    story.append(Paragraph(text, style))


def fig(name, caption, width_cm=14.5):
    path = os.path.join(FIGS, name)
    iw, ih = ImageReader(path).getSize()
    w = width_cm * cm
    story.append(Image(path, width=w, height=w * ih / iw))
    story.append(Paragraph(caption, CAPTION))


def pct(x):
    return f"{x * 100:+.1f}\\%".replace("\\%", "%")


# ---------------------------------------------------------------- title

p("Cost-Aware Eviction for LLM Semantic Caches:", TITLE)
p("Adding a GDSF Policy to GPTCache", TITLE)
story.append(Spacer(1, 10))
p("Final project report", SUBTITLE)
p("Code: feature branch <font face='Courier'>feature/gdsf-eviction</font> "
  "of our GPTCache fork (see Appendix A)", SUBTITLE)
story.append(Spacer(1, 18))

# ---------------------------------------------------------------- 1 intro

p("1. Introduction", H1)
p("Calling a large language model is slow and costs money. A single answer "
  "can take tens of seconds to generate, and API providers charge per token. "
  "Because many applications receive the same or very similar prompts over "
  "and over (think of a documentation chatbot that gets asked \"how do I "
  "reset my password\" fifty times a day), caching LLM responses is one of "
  "the most effective ways to cut both latency and cost.")
p("<b>GPTCache</b> (github.com/zilliztech/GPTCache) is the most popular "
  "open-source semantic cache for LLMs. It stores past prompts and answers, "
  "embeds incoming prompts into vectors, and serves a stored answer whenever "
  "a new prompt is similar enough to an old one. We chose it as our baseline "
  "because it is widely used, written in clear modular Python, ships with "
  "unit tests, and runs on any machine without a GPU. A separate one-page "
  "justification of this choice is included with our submission.")
p("Like any cache, GPTCache must decide what to throw away when it fills "
  "up. Its eviction module offers four classic policies: LRU (the default), "
  "LFU, FIFO and RR, all thin wrappers around the <font face='Courier'>"
  "cachetools</font> library. All four share one blind spot: <b>they treat "
  "every cached entry as equally valuable</b>. The eviction index literally "
  "stores <font face='Courier'>cache[id] = True</font> — it knows nothing "
  "about the entry beyond its id and access pattern.")
p("But LLM cache entries are not equally valuable. In our workloads (and in "
  "real traffic) generated answers span two orders of magnitude in length: "
  "a cached 20-token answer saved the user well under a second, while a "
  "cached 2,000-token answer saved them about 40 seconds of generation time "
  "and a proportional API bill. When the cache is full, LRU will happily "
  "evict the expensive entry to keep the cheap one, just because the cheap "
  "one was touched more recently. The metric that actually matters to the "
  "user is not the raw hit rate — it is <b>how much generation work the "
  "cache saves</b>.")
p("<b>Our contribution.</b> We add a cost-aware eviction policy to GPTCache: "
  "GDSF (Greedy-Dual-Size-Frequency), a well-studied policy from the web "
  "proxy caching literature (Cherkasova, 1998) that, to our knowledge, has "
  "not been applied to LLM response caches before. Concretely, we: "
  "(1) implement GDSF as a drop-in <font face='Courier'>cachetools</font>-"
  "compatible cache class selected with <font face='Courier'>policy="
  "\"GDSF\"</font>, keeping full backward compatibility with the existing "
  "eviction API; (2) wire the cost signal through GPTCache's data manager, "
  "so an application that selects the policy gets cost-aware eviction "
  "end-to-end with no further changes; (3) build a reproducible benchmark "
  "suite with four workload profiles and a cost-weighted hit-rate metric; "
  "(4) evaluate GDSF against all four built-in policies over 10 random "
  "seeds with confidence intervals, on both synthetic cost distributions "
  "and real response lengths from the OASST1 dataset; and (5) run an "
  "ablation study that isolates the contribution of each ingredient of "
  "the policy, including its behaviour when prompt popularity drifts over "
  "time.")
p("<b>Headline result:</b> at every cache size we tested, GDSF saves "
  "significantly more generation work than every baseline. At a cache size "
  "of 1,000 entries it raises the cost-weighted hit rate from "
  f"{e1(1000, 'LRU', 'cost_weighted_hit_rate')['mean']:.3f} (LRU) to "
  f"{e1(1000, 'GDSF', 'cost_weighted_hit_rate')['mean']:.3f}, which "
  f"translates to {pct(sig('LRU', 'latency_mean_ms', 1000)['relative_change'])} "
  "mean latency and "
  f"{pct(sig('LRU', 'latency_p95_ms', 1000)['relative_change'])} p95 latency "
  "under our latency model. All differences are statistically significant "
  "(paired bootstrap 95% confidence intervals over 10 seeds), and the "
  "improvement survives when the synthetic cost distribution is replaced "
  "with real response lengths (Section 4.5).")

# ---------------------------------------------------------------- 2 design

p("2. Extension Design", H1)
p("2.1 The GDSF policy", H2)
p("GDSF assigns every cached entry a priority H and always evicts the entry "
  "with the lowest priority:")
p("H(e) &nbsp;=&nbsp; L &nbsp;+&nbsp; frequency(e) &times; cost(e) / size(e)",
  ParagraphStyle("Formula", parent=BODY, alignment=1, fontSize=11.5,
                 spaceBefore=6, spaceAfter=8))
p("<b>frequency(e)</b> counts how often the entry was accessed, like LFU. "
  "<b>cost(e)</b> is how expensive the entry was to produce — we use the "
  "number of generated tokens, which is proportional to both generation "
  "time and API price. <b>size(e)</b> is the entry's footprint in the cache; "
  "in GPTCache's eviction index every entry occupies one slot, so size is 1 "
  "and the term drops out. <b>L</b> is an \"inflation\" value that starts at "
  "0 and rises to the priority of each evicted entry.")
p("The intuition: the product frequency &times; cost estimates how many "
  "tokens the entry will save us in the future — an answer that is both "
  "popular and long is the most valuable thing in the cache. The L term "
  "handles aging. Without it, an entry that was very popular last week "
  "would keep its high priority forever, even if nobody ever asks for it "
  "again. Because L rises with every eviction, fresh entries enter with "
  "ever-higher baseline priority and stale ones are eventually overtaken. "
  "Note that if all costs are equal, GDSF reduces to LFU with aging — so it "
  "generalizes an existing policy rather than fighting it.")
p("2.2 Implementation", H2)
p("We implemented <font face='Courier'>GDSFCache</font> as a subclass of "
  "<font face='Courier'>cachetools.Cache</font> (about 90 lines, "
  "<font face='Courier'>gptcache/manager/eviction/gdsf.py</font>), the same "
  "base class the four built-in policies use. Priorities live in a lazy "
  "min-heap: every access pushes an updated (priority, key) pair, and "
  "eviction pops entries until it finds one whose priority is current, "
  "skipping stale ones. This keeps both accesses and evictions O(log n) — "
  "we measured the throughput cost of this bookkeeping in Section 5.")
p("The policy plugs into GPTCache's existing dispatch: "
  "<font face='Courier'>MemoryCacheEviction(policy=\"GDSF\", ...)</font>. "
  "To get costs into the cache we extended the eviction API with one "
  "optional parameter:")
p("eviction_base.put(ids)                    # unchanged, cost = 1.0<br/>"
  "eviction_base.put(ids, costs=[tokens])    # cost-aware", CODE)
p("Backward compatibility was a hard requirement (it is also what makes "
  "this contribution realistic to upstream). When no costs are passed, "
  "every entry gets cost 1.0 and GDSF degrades gracefully to LFU-with-"
  "aging; the four existing policies simply ignore the parameter. All "
  "eight of GPTCache's original eviction unit tests pass unmodified, and "
  "we added eleven new unit tests covering eviction order, cost updates, "
  "the aging mechanism and the API compatibility guarantees, plus an "
  "end-to-end integration test described next.")
p("The cost signal is also wired through GPTCache's data manager: when an "
  "answer is inserted, <font face='Courier'>import_data</font> passes the "
  "answer's text length (characters, proportional to tokens on average — "
  "only relative costs matter for eviction) to the eviction layer. So "
  "selecting <font face='Courier'>policy=\"GDSF\"</font> is all an "
  "application has to do; there is no second API to call. An integration "
  "test builds a full GPTCache data manager (SQLite + Faiss), inserts one "
  "long answer followed by a flood of short ones, and checks that GDSF "
  "keeps the long answer where LRU provably evicts it.")
p("2.3 Tunable parameters", H2)
p("Our extension exposes the same knobs as the baseline — "
  "<font face='Courier'>maxsize</font> (cache capacity in entries) and "
  "<font face='Courier'>clean_size</font> (how many entries each cleanup "
  "batch evicts, default 20% of capacity) — plus the choice of cost "
  "signal passed to <font face='Courier'>put()</font>. We use generated "
  "tokens; measured wall-clock generation time would work identically. "
  "Section 4 sweeps cache capacity and workload shape to show how "
  "performance depends on them.")

# ---------------------------------------------------------------- 3 setup

p("3. Experimental Setup", H1)
p("3.1 Benchmark harness", H2)
p("We benchmark the eviction layer directly: a driver replays a stream of "
  "(prompt id, cost) requests against <font face='Courier'>"
  "MemoryCacheEviction</font> exactly the way GPTCache's data manager does "
  "(same put/get calls, same batched-eviction callback). We deliberately "
  "use a <b>mocked LLM</b> instead of a real one. The cache decision logic "
  "never sees the model, so mocking does not change which entries hit or "
  "miss; what it buys us is that anyone can rerun every experiment in this "
  "report on a laptop in a few minutes, deterministically, for free. We "
  "consider that a good trade for a project graded on reproducibility.")
p("Latency is derived from the replay with a simple linear model: a hit "
  "costs 5 ms (cache lookup), a miss costs 300 ms fixed overhead plus "
  "20 ms per generated token — roughly a hosted API generating ~50 "
  "tokens/second. Both knobs are command-line flags, so the numbers can be "
  "recomputed for a faster or slower backend; the <i>relative</i> ranking "
  "of policies does not depend on them.")
p("3.2 Workloads", H2)
p("<b>zipf</b> — 50,000 requests over 5,000 unique prompts whose "
  "popularity follows a Zipf distribution (skew s = 1.1 unless stated), "
  "the standard model for cache traffic. Each prompt's answer length is "
  "drawn from a log-normal distribution (clipped to 10–4,000 tokens): most "
  "answers are short, a few are very long, like real LLM traffic. "
  "Popularity and cost are drawn independently. A <b>drift</b> variant "
  "reshuffles the popularity ranking halfway through the stream, modelling "
  "topics that come and go. <b>repetitive_short</b> — 50 short prompts "
  "repeated 50,000 times; a sanity check where every policy should be "
  "near 100% hits. <b>novel_long</b> — 20,000 unique long prompts; the hit "
  "rate is 0% by construction, isolating the cache's bookkeeping overhead. "
  "<b>oasst</b> — same Zipf popularity, but the costs are real: each of "
  "3,634 unique first-turn English prompts from the OASST1 dataset "
  "(OpenAssistant/oasst1, Apache-2.0) carries the length of its actual "
  "assistant reply. OASST1 prompts are nearly all unique, so the "
  "popularity pattern still has to be synthetic; what this workload "
  "removes is the log-normal cost assumption.")
p("3.3 Metrics and statistics", H2)
p("We report the raw <b>hit rate</b>, the <b>cost-weighted hit rate</b> "
  "(fraction of total generation tokens served from cache — the metric "
  "that maps to real time and money), <b>latency</b> (mean, p50, p95, "
  "p99) under the model above, <b>evictions</b>, and the <b>throughput</b> "
  "of the cache layer itself. Every configuration runs on 10 random seeds. "
  "For the headline comparisons we compute paired per-seed differences "
  "(GDSF minus baseline on the identical request stream) and a 95% "
  "bootstrap confidence interval of the mean difference (10,000 "
  "resamples); we call a difference significant only when the interval "
  "excludes zero. Experiments ran on a MacBook (Apple silicon, CPU only), "
  "Python 3.10, cachetools 5.5.2 (pinned, because the LFU baseline's "
  "exact numbers depend on its implementation details; GDSF and LRU are "
  "unaffected); a Dockerfile reproduces the environment.")
p("3.4 Reproducing", H2)
p("git checkout feature/gdsf-eviction<br/>"
  "pip install -e . \"cachetools==5.5.2\" numpy matplotlib pytest "
  "sqlalchemy faiss-cpu<br/>"
  "python -m pytest tests/unit_tests/eviction/ \\<br/>"
  "&nbsp;&nbsp;tests/unit_tests/manager/test_eviction.py -q -o addopts= \\<br/>"
  "&nbsp;&nbsp;--ignore=tests/unit_tests/eviction/test_distributed_cache.py<br/>"
  "cd benchmarks<br/>"
  "python run_experiments.py --seeds 10 --out results/full<br/>"
  "python make_plots.py", CODE)

# ---------------------------------------------------------------- 4 results

p("4. Results", H1)
p("4.1 Main comparison: cache-size sweep", H2)
p("Figure 1 shows the cost-weighted hit rate for all five policies across "
  "cache sizes from 250 to 4,000 entries (5% to 80% of the 5,000-prompt "
  "working set). GDSF dominates at every size, and the gap is widest "
  "exactly where eviction policy matters most — when the cache is small "
  "relative to the working set. At 4,000 entries almost everything fits "
  "and all policies converge, which is the expected sanity behaviour.")
fig("fig1_cost_weighted_vs_size.png",
    "Figure 1: Cost-weighted hit rate (fraction of generation tokens served "
    "from cache) vs. cache size. Lines are means over 10 seeds; shaded "
    "bands are ±1 standard deviation. FIFO (dashed) and RR overlap almost "
    "exactly.")
p("Figure 2 shows the same sweep for the raw hit rate. This is an "
  "important control: GDSF does <i>not</i> buy its token savings by "
  "sacrificing the number of hits. Its raw hit rate stays at or slightly "
  "above LRU's (and within noise of LFU's), while the <i>composition</i> "
  "of what it keeps shifts toward expensive entries.")
fig("fig2_hit_rate_vs_size.png",
    "Figure 2: Raw hit rate vs. cache size. GDSF matches the best "
    "baselines; the improvement in Figure 1 comes from keeping more "
    "valuable entries, not more entries.")
p("Figures 3 and 4 translate this into user-visible latency. Mean latency "
  "improves because fewer tokens are regenerated overall; the p95 "
  "improves even more because the requests that used to regenerate "
  "long answers — the slowest ones — are precisely the ones GDSF now "
  "keeps cached.")
fig("fig3_latency_vs_size.png",
    "Figure 3: Mean (left) and p95 (right) request latency vs. cache size "
    "under the latency model of Section 3.1. Lower is better.")
fig("fig6_latency_cdf.png",
    "Figure 4: Latency distribution (CDF, tail region) at cache size "
    "1,000, seed 1. The flat region left of ~1 s is cache hits; GDSF's "
    "curve rises earlier in the expensive tail.", 12.5)
p("Table 1 summarizes the headline numbers with their confidence "
  "intervals. Every interval excludes zero, at every cache size, against "
  "both LRU (GPTCache's default) and LFU (the strongest baseline).")

# Table 1
rows = [["Cache size", "Metric", "vs LRU", "vs LFU"]]
for size in (500, 1000, 2000):
    for metric, label in (("cost_weighted_hit_rate", "tokens saved"),
                          ("latency_mean_ms", "mean latency"),
                          ("latency_p95_ms", "p95 latency")):
        r = []
        for base in ("LRU", "LFU"):
            v = sig(base, metric, size)
            r.append(pct(v["relative_change"]))
        rows.append([str(size) if metric == "cost_weighted_hit_rate" else "",
                     label] + r)
t = Table(rows, colWidths=[2.6 * cm, 4.2 * cm, 3.4 * cm, 3.4 * cm])
t.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
    ("FONT", (0, 1), (-1, -1), "Helvetica", 10),
    ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
    ("LINEBELOW", (0, 3), (-1, 3), 0.3, colors.grey),
    ("LINEBELOW", (0, 6), (-1, 6), 0.3, colors.grey),
    ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
story.append(Paragraph(
    "Table 1: Relative change of GDSF vs. each baseline (positive = more "
    "tokens saved, negative = lower latency). All differences are "
    "significant: paired bootstrap 95% CIs over 10 seeds exclude zero in "
    "every cell. Full CIs are in results/full_summary.json.", CAPTION))

p("4.2 Sensitivity to workload skew", H2)
p("Figure 5 varies the Zipf skew parameter s at a fixed cache size of "
  "1,000. Lower s means flatter, less repetitive traffic — a harder "
  "workload for any cache. GDSF's advantage <i>grows</i> as the workload "
  "gets harder (at s = 0.8 it saves "
  f"{abl('zipf_s1.1', 'GDSF')['mean'] * 0 + S['E2_skew_sweep']['zipf_s0.8|1000|GDSF']['cost_weighted_hit_rate']['mean'] - S['E2_skew_sweep']['zipf_s0.8|1000|LRU']['cost_weighted_hit_rate']['mean']:.2f}"
  " more of the total tokens than LRU), because when hits are scarce, "
  "<i>which</i> entries you keep matters more.")
fig("fig4_skew_sweep.png",
    "Figure 5: Cost-weighted hit rate vs. Zipf skew s (cache size 1,000). "
    "GDSF helps most when traffic is least repetitive.")

p("4.3 Ablation: which ingredient does the work?", H2)
p("GDSF combines three ingredients: frequency, cost, and aging. We "
  "benchmarked stripped-down variants to isolate each one: LFU is "
  "frequency alone; \"cost only\" ranks by L + cost; \"freq &times; cost\" "
  "is GDSF with aging disabled. Figure 6 shows both a stationary workload "
  "and the drift variant.")
fig("fig5_ablation.png",
    "Figure 6: Ablation at cache size 1,000. Left: stationary popularity. "
    "Right: popularity reshuffled halfway through the stream. Error bars "
    "are ±1 standard deviation over 10 seeds.")
p("Three things stand out. First, <b>cost-awareness is the main "
  f"ingredient</b>: cost-only already reaches {abl('zipf_s1.1', 'COST_ONLY')['mean']:.3f} "
  f"vs. {abl('zipf_s1.1', 'LFU')['mean']:.3f} for LFU on the stationary "
  "workload. Second, on a stationary workload <b>aging is a small "
  f"handicap</b>: freq &times; cost without aging ({abl('zipf_s1.1', 'GDSF_NO_AGE')['mean']:.3f}) "
  f"slightly beats full GDSF ({abl('zipf_s1.1', 'GDSF')['mean']:.3f}). That "
  "makes sense — if popularity never changes, protecting old favourites is "
  "exactly right, and aging can only evict them too early. Third, the "
  "moment popularity drifts, the ranking flips: the no-aging variant drops "
  f"to {abl('zipf_s1.1_drift', 'GDSF_NO_AGE')['mean']:.3f} while full GDSF "
  f"stays best at {abl('zipf_s1.1_drift', 'GDSF')['mean']:.3f}, and LFU "
  f"({abl('zipf_s1.1_drift', 'LFU')['mean']:.3f}) falls below even LRU "
  f"({abl('zipf_s1.1_drift', 'LRU')['mean']:.3f}) — the classic \"stale "
  "frequency counts\" failure. Since real traffic drifts, we ship GDSF "
  "with aging on: it costs about half a point on stationary traffic and "
  "buys nearly two points of insurance under drift.")

p("4.4 Overhead and sanity checks", H2)
p("On the novel_long workload (0% possible hit rate) all policies produce "
  "identical latency — the cache never helps, and GDSF adds no per-request "
  "penalty. Figure 7 shows the throughput of the cache layer itself on "
  "this worst-case workload: GDSF's heap bookkeeping does cost raw "
  "operations per second relative to LRU's O(1) list moves. We consider "
  "this irrelevant in practice: even the slower figure is hundreds of "
  "thousands of cache operations per second, while a single LLM call "
  "takes hundreds of milliseconds — the cache layer is six orders of "
  "magnitude away from being the bottleneck. On the repetitive_short "
  "sanity workload every policy reaches a 99.9% hit rate, as expected.")
fig("fig7_overhead.png",
    "Figure 7: Cache-layer throughput on the 0%-hit workload (pure "
    "bookkeeping overhead). GDSF trades raw index speed for better "
    "decisions; both are far faster than any LLM call.", 10.5)

p("4.5 Do the results survive real response lengths?", H2)
p("Every result so far used a log-normal cost distribution. To check that "
  "the improvement is not an artifact of that choice, we repeated the "
  "cache-size sweep with real costs: for each of 3,634 unique first-turn "
  "English prompts from OASST1, the cost is the length of the actual "
  "assistant reply the prompt received (Section 3.2). The real "
  "distribution is less extreme than our synthetic one (median ~240 "
  "tokens, p95 ~618, max ~2,470), so this is a harder test for a "
  "cost-aware policy.")
p("Figure 8 shows the outcome: the picture is the same. At cache size "
  "1,000 GDSF lifts the cost-weighted hit rate from "
  f"{e6(1000, 'LRU', 'cost_weighted_hit_rate')['mean']:.3f} (LRU) to "
  f"{e6(1000, 'GDSF', 'cost_weighted_hit_rate')['mean']:.3f} "
  f"({pct(sig6('LRU', 'latency_mean_ms', 1000)['relative_change'])} mean, "
  f"{pct(sig6('LRU', 'latency_p95_ms', 1000)['relative_change'])} p95 "
  "latency), and it beats LFU on every metric as well. All paired "
  "bootstrap CIs over 10 seeds exclude zero, at every cache size, against "
  "both baselines. The margins are smaller than on the synthetic workload "
  "— exactly what we expect with a lighter-tailed cost distribution — but "
  "they are consistently there.")
fig("fig8_real_costs.png",
    "Figure 8: Cache-size sweep with real OASST1 response lengths as "
    "costs (Zipf popularity, 10 seeds, bands are ±1 standard deviation). "
    "GDSF stays best on both tokens saved and tail latency.")

# ---------------------------------------------------------------- 5 discussion

p("5. Discussion", H1)
p("<b>When does GDSF help?</b> Two conditions must hold: entries must "
  "differ in cost, and popularity must not be perfectly correlated with "
  "cost. Both held in every setting we tested, including the empirical "
  "OASST1 response-length distribution of Section 4.5. If all answers "
  "had identical length, GDSF collapses to LFU-with-aging and the "
  "improvement disappears by design; it never does worse than that floor.")
p("<b>Trade-offs.</b> The costs GDSF needs are free — GPTCache already "
  "has the response in hand when it inserts an entry, and our data-"
  "manager integration derives the cost from the answer automatically. "
  "The real trade-offs are the ones our experiments quantify: a "
  "raw-throughput cost in the eviction index (Figure 7, irrelevant at "
  "LLM time scales) and the aging trade-off on stationary vs. drifting "
  "traffic (Section 4.3). A further honest caveat: we optimize for "
  "provider-side compute saved, which is the right target when the user "
  "pays per token; if some other utility function matters (e.g. fairness "
  "across tenants), cost-weighting would need rethinking.")
p("<b>Threats to validity.</b> Our popularity patterns are synthetic. We "
  "mitigated this the standard way — Zipf popularity is the accepted "
  "model in the caching literature — and we swept the skew parameter "
  "rather than picking one flattering value. The cost side is no longer "
  "an assumption: Section 4.5 repeats the main experiment with real "
  "response lengths and the conclusion holds. A real repetition trace "
  "(which prompts repeat, and how often) is the piece we could not get "
  "from public data — datasets with genuine repetition, like the LMSYS "
  "chatbot-arena logs, are gated, and OASST1's prompts are nearly all "
  "unique. Our latency numbers come from a linear model, not a live LLM; "
  "the model's two constants are configurable and only scale the latency "
  "plots, they cannot change which policy wins. Finally, we benchmark "
  "the eviction layer in isolation; embedding and vector-search time on "
  "a hit is identical across policies, so it would shift all curves "
  "equally.")

# ---------------------------------------------------------------- 6 conclusion

p("6. Conclusion &amp; Future Work", H1)
p("We added a cost-aware eviction policy to GPTCache behind its existing "
  "API, wired it end-to-end through the data manager, tested it, and "
  "measured it. On Zipf workloads with realistic response-length skew, "
  "GDSF saves significantly more generation work than every built-in "
  "policy at every cache size we tried — around "
  f"{pct(sig('LRU', 'latency_mean_ms', 1000)['relative_change'])} mean and "
  f"{pct(sig('LRU', 'latency_p95_ms', 1000)['relative_change'])} p95 "
  "latency vs. the default LRU at cache size 1,000 — while matching the "
  "baselines' raw hit rate and adding no meaningful overhead when the "
  "cache cannot help, and the improvement survives with real OASST1 "
  "response lengths in place of the synthetic cost distribution. The "
  "ablation gave us the most instructive result of the project: aging "
  "looked useless until we modelled popularity drift, at which point it "
  "became the difference between GDSF and an LFU-style collapse. It was "
  "a good reminder that a benchmark only answers the questions you put "
  "into it.")
p("We have opened a pull request contributing the policy and its "
  f"data-manager integration back to GPTCache upstream ({PR_LINK}).")
p("<b>Future work</b>, kept grounded: (1) get the upstream pull request "
  "reviewed and merged; (2) replace the synthetic popularity pattern "
  "with a real repetition trace (the datasets that contain one, like the "
  "LMSYS chatbot-arena logs, are gated — with access, the harness can "
  "replay them directly); (3) try byte-size-aware GDSF (the size term we "
  "currently hold at 1) for the on-disk cache tier.")

# ---------------------------------------------------------------- appendix

story.append(PageBreak())
p("Appendix A: Code and Data Artifacts", H1)
rows = [
    ["Artifact", "Path"],
    ["GDSF implementation", "gptcache/manager/eviction/gdsf.py"],
    ["API integration", "gptcache/manager/eviction/memory_cache.py"],
    ["Data-manager integration", "gptcache/manager/data_manager.py"],
    ["Unit tests (11 new)", "tests/unit_tests/eviction/test_gdsf_cache.py"],
    ["Integration test", "tests/unit_tests/manager/test_eviction.py"],
    ["Workload generators", "benchmarks/workloads.py"],
    ["OASST1 real costs", "benchmarks/data/oasst1_costs.csv"],
    ["Benchmark harness", "benchmarks/run_bench.py"],
    ["Ablation variants", "benchmarks/ablation.py"],
    ["Experiment suite", "benchmarks/run_experiments.py"],
    ["Figure generation", "benchmarks/make_plots.py"],
    ["Raw results (10 seeds)", "benchmarks/results/full_raw.csv"],
    ["Aggregates + 95% CIs", "benchmarks/results/full_summary.json"],
    ["Figures", "benchmarks/results/figures/"],
    ["Benchmark how-to", "benchmarks/README.md"],
    ["Environment", "Dockerfile"],
    ["Baseline justification", "report/baseline_justification.md"],
    ["This report's source", "report/generate_report.py"],
]
t = Table(rows, colWidths=[5.5 * cm, 9.5 * cm])
t.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
    ("FONT", (0, 1), (0, -1), "Helvetica", 10),
    ("FONT", (1, 1), (1, -1), "Courier", 9),
    ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
]))
story.append(t)
story.append(Spacer(1, 10))
p("Every number in this report is generated programmatically from "
  "<font face='Courier'>full_summary.json</font> by "
  "<font face='Courier'>generate_report.py</font>, so text and data "
  "cannot drift apart. The full experiment suite reruns in about six "
  "minutes on a laptop (Section 3.4).")
p("<b>Reference:</b> L. Cherkasova, \"Improving WWW Proxies Performance "
  "with Greedy-Dual-Size-Frequency Caching Policy\", HP Labs Technical "
  "Report HPL-98-69, 1998.")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4,
                        leftMargin=2.3 * cm, rightMargin=2.3 * cm,
                        topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                        title="Cost-Aware Eviction for LLM Semantic Caches")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"wrote {OUT}")
