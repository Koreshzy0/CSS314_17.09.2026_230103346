# ANALYSIS.md — Cache Line & False Sharing Mechanics

**Run environment:** CPU Apple M2, 8 logical cores, detected cache line 128 B, Numba 0.68.0, affinity pinning not supported (macOS), 4 threads x 60,000,000 updates, median of 3 runs. GIT_USERNAME = Koreshzy0, INSTANCE_SEED = 9794293981315474915.

## Question 1 (MESI Protocol State Transitions)

Stride = 1: `buffer[0]` and `buffer[1]` are 8 B apart, so they live in the **same** cache line. Assume Core 0 already owns the line in **M** (it has been storing) and Core 1 holds it **I**.
1. Core 1 executes `store buffer[1]`; its L1 misses (line is I) → it issues a **Read-For-Ownership (RFO)** on the interconnect.
2. Core 0 snoops the RFO: its copy is **M**, so it supplies/writes back the dirty line and goes **M → I**.
3. Core 1 receives the line, installs it (**I → E → M** after its store commits).
4. Core 0 now stores `buffer[0]`: its copy is I → RFO → Core 1 goes **M → I**, Core 0 gets it **M**. This repeats on every iteration (ping-pong), although the two cores never touch the same address.

(If the line was **S** in both caches, the store first needs an upgrade/invalidate: S → M on the writer, S → I on the peer.)

The RFO stalls Core 1's store buffer because stores must commit to L1 in program order and a store can only commit when the line is held in M or E. The head store waits for ownership; younger stores queue behind it, the store buffer fills up, and the pipeline stalls.

Measured effect: 2.702 ns/update for stride 1 vs 2.559 ns/update for stride 8 (2.549 ns for stride 16). On my Apple M2 the effect is small (about 5.6%), see Q2 for why I think so.

## Question 2 (Quantitative Slowdown & Coherency Overhead)

| Layout | stride (B) | elapsed (ns) | ns / update | distinct lines (64 B / detected 128 B) |
|---|---|---|---|---|
| A contiguous | 1 (8) | 162,123,667 | 2.702 | 1 / 1 |
| B half-line | 4 (32) | 162,256,416 | 2.704 | 2 / 1 |
| C line-padded | 8 (64) | 153,518,208 | 2.559 | 4 / 2 |
| D prefetch-safe | 16 (128) | 152,948,334 | 2.549 | 4 / 4 |
| private (register) | 8 | 17,206,125 | 0.287 | 4 / 2 |

ns/update = elapsed_ns / 60,000,000 (average latency of one thread's update; the 4 threads run concurrently).

Slowdown stride 1 vs stride 8 = 162,123,667 / 153,518,208 = **1.056x** (matches `slowdown_stride1_vs_stride8` in the manifest). Stride 1 vs private = **9.422x**; stride 8 vs private = **8.922x**.

The private baseline removes all per-iteration memory stores, so (stride 8 − private) = 2.559 − 0.287 ≈ 2.27 ns is the cost of the L1 store-to-load forwarding dependency chain, and (stride 1 − stride 8) = 2.702 − 2.559 ≈ 0.14 ns is the cross-core coherence cost I could measure.

So on this machine almost all of the cost (about 2.27 of 2.70 ns) comes from the dependent load/store chain itself, and only about 5% from cross-core contention. I did not get the dramatic slowdown that false sharing shows on many x86 CPUs. Possible reasons (I did not test them separately): the M2 performance cores of one cluster share a large L2, so moving a line between cores is cheap; the already long store-forwarding chain per update hides part of the coherence delay; and without affinity pinning macOS decides on which cores the 4 threads run.

Since the detected line is 128 B, layouts A, B and C all put threads on shared 128 B lines (distinct 128 B lines: 1, 1, 2), and only D (stride 16) gives each thread its own line (4). The numbers agree with this grouping: A ≈ B (162.1 / 162.3 ms, both one 128 B line) and C ≈ D (153.5 / 152.9 ms), but the gap between the two groups is small.

## Question 3 (L2 Spatial Prefetcher & 128-Byte Pairing)

Measured stride 8 = 2.559 ns vs stride 16 = 2.549 ns (ratio 1.004x). On Intel x86 the L2 adjacent-cache-line ("spatial") prefetcher fetches the buddy line of a 128 B-aligned pair (even/odd 64 B lines) together with the demanded line, so two 64 B lines behave like one 128 B sector. With 64 B padding (stride 8) thread t's counter and thread t+1's counter sit in different 64 B lines but possibly in the same 128 B pair: the prefetcher pulls the neighbour's line into the core, and the neighbour's next store invalidates it again (secondary invalidations and extra RFO traffic). 128 B spacing (stride 16) puts every counter in its own pair, which removes this.

Padding of only 64 B still suffers when the line size is really 128 B (Apple Silicon, detected 128 B here: with stride 8, threads 0 and 1 share one 128 B line and threads 2 and 3 another, lines128 = 2) or when the prefetcher is enabled and the counters fall in the same aligned 128 B pair.

In my run the ratio is ≈ 1 (1.004x), so the prefetcher / 128 B pairing effect is **not visible** on my M2: stride 8 and stride 16 are practically equal, even though stride 8 shares 128 B lines in pairs. I can say that the effect exists in theory, but my measurement does not show it.

## Question 4 (Compiler Register Promotion vs. Volatile Memory Stores)

Numba's LLVM IR for `worker_strided_accumulate` (`inspect_llvm()`, compiled body only) contains **1** `store i64` and **1** `load i64` instruction and no Python object calls (`python_object_fallback = False`, `audit_passed = True`).

An optimizing compiler would normally apply scalar promotion (LICM): load `buffer[slot]` once before the loop, keep the running sum in a register and store it once after the loop, because nothing else in the loop can observe the intermediate values (Numba cannot mark the access `volatile`/atomic).

My loop prevents this: the address of each access is `idx = slot + (nv & 1)`, where `nv` is the value just computed from memory, so the next load/store address is data-dependent on the previous store. The value is always even, so `idx == slot` at runtime, but the compiler cannot prove it, therefore it must keep a real load and store in each iteration.

Evidence from timing: stride 8 (one thread per 64 B line) = 2.559 ns/update vs the private register baseline = 0.287 ns/update; the extra 8.92x (about 2.27 ns per update) is the cost of real L1 stores plus store-to-load forwarding.

## Question 5 (CPython GIL Masking vs. nogil=True Execution)

With plain Python threads only the thread holding the GIL executes bytecode, so two stores to the same cache line are never issued by two cores at the same moment; the hardware never sees concurrent RFO ping-pong, so false sharing cannot show up. What happens instead is **software lock contention**: threads fight for the GIL, each handoff costs a condition-variable wakeup plus a context switch, and bytecode is interleaved, not parallel.

My measurement (`gil_demo` in the manifest, 8,000,000 pure-Python updates): 1 thread = 173,098,500 ns, 4 threads = 178,906,958 ns (ratio 1.034x). The 4 threads are not faster, and are slightly slower (about 3.4%), although the work is the same. So the GIL gives no parallel speedup and adds a small switching overhead.

With `@njit(nogil=True)` the GIL is released for the native loop, 4 threads really run on 4 cores, and the slowdown of stride 1 is then caused purely by hardware cache-line contention (RFO traffic, 1.056x slower than stride 8 on my M2), with no lock involved.
