# Task 19 — Monte Carlo: the ideal weak-scaling workload

## Where does the non-zero serial fraction α come from?
| Source | Typical cost model | Scaling | Remedy |
|---|---|---|---|
| PRNG seed management | master generates P seeds / jump-ahead state sequentially: t_seed·P | O(P) | counter-based RNG (Philox/Threefry) or hash(seed, rank): O(1), reproducible |
| Statistical coupling of streams | leapfrog/skip-ahead and stream correlation checks | O(1)–O(P) | independent substreams (SplitMix/SeedSequence.spawn) |
| Reduction of partial estimates | tree: ⌈log₂P⌉·(L + m/B); flat (to root): (P−1)·L | O(log P) vs O(P) | tree/Rabenseifner reduce, hierarchical (node → rack) |
| Output serialization | rank-0 collects/writes P records: t_out·P | O(P) | parallel I/O, write only aggregated statistics |
| Start-up/finalize (MPI_Init, load config) | job launcher cost | O(1)–O(log P) | constant at fixed α |

## Impact of the reduction tree on S(P)
Compute per rank is constant (T₀ = 10 s). Non-scaling overhead O(P) (seconds) turns the classical S = P − α(P−1)
into S_P = P·T₀/(T₀ + overhead(P)), i.e. α_eff(P) = overhead/(T₀+overhead). With tree reduction overhead(P) = ⌈log₂P⌉·L:
at P = 2²⁰ it is only 20·L = 0.40 ms, α_eff ≈ 4.0e-05: **a tree reduction costs essentially nothing**
and Monte Carlo scales almost perfectly (S ≈ P).

## Numeric model (L = 20 μs per tree level, 5 μs per message for the flat reduce, t_seed = 1 μs, t_out = 50 μs per rank)

Parallel efficiency E = T₀/(T₀ + overhead):

| P | tree overhead [ms] | E tree only | E flat reduce | E tree + serial seeding | E tree + serial output | E tree + seeding + output |
|---|---|---|---|---|---|---|
| 1 | 0.000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.99999 |
| 4 | 0.040 | 1.00000 | 1.00000 | 1.00000 | 0.99998 | 0.99998 |
| 16 | 0.080 | 0.99999 | 0.99999 | 0.99999 | 0.99991 | 0.99991 |
| 64 | 0.120 | 0.99999 | 0.99997 | 0.99998 | 0.99967 | 0.99966 |
| 256 | 0.160 | 0.99998 | 0.99987 | 0.99996 | 0.99871 | 0.99868 |
| 1,024 | 0.200 | 0.99998 | 0.99949 | 0.99988 | 0.99489 | 0.99478 |
| 4,096 | 0.240 | 0.99998 | 0.99796 | 0.99957 | 0.97991 | 0.97951 |
| 16,384 | 0.280 | 0.99997 | 0.99188 | 0.99834 | 0.92426 | 0.92286 |
| 65,536 | 0.320 | 0.99997 | 0.96827 | 0.99346 | 0.75318 | 0.74948 |
| 262,144 | 0.360 | 0.99996 | 0.88412 | 0.97442 | 0.43276 | 0.42790 |
| 1,048,576 | 0.400 | 0.99996 | 0.65604 | 0.90506 | 0.16018 | 0.15754 |

**Reading.** The log-depth tree keeps E > 0.9999 up to 10⁶ ranks. The O(P) terms are the real danger: sequential
seeding costs ≈1 s and rank-0 output ≈ 52 s at 2²⁰ ranks (E ≈ 0.16). Lesson: for embarrassingly-parallel
codes, α is dominated by O(P) bookkeeping, not by the arithmetic or the reduction tree; replacing it with O(1)/O(log P)
mechanisms restores Gustafson-ideal scaling.
