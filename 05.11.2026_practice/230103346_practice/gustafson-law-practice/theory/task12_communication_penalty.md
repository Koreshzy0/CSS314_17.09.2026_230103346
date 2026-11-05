# Task 12 — Gustafson Model with Communication Overhead C(P)

## Model
Take the parallel run of the *baseline* (communication-free) schedule as one unit, s + p = 1, and add the time the
machine must spend communicating, which grows with P:

$$T_P = s + p + C(P) = 1 + C(P),\qquad C(P)=\gamma\,f(P).$$

The scaled workload is unchanged (work still s + pP in single-core units), so

$$\boxed{S_C(P)=\frac{s+pP}{1+C(P)}=\frac{P-\alpha(P-1)}{1+C(P)}},\qquad E_C(P)=\frac{1-\alpha+\alpha/P}{1+C(P)}.$$

Equivalent "effective serial fraction" form: S_C = P − α_eff(P)(P−1) with
α_eff = (P − S_C)/(P−1) = (S_C is measured in Task 23 and α_eff(P) is fitted from it).
For C = 0 the classical law is recovered.

## Choices of f(P)
* f(P) = log₂ P — tree collectives / reductions (latency term L·log P).
* f(P) = √P — 2D nearest-neighbor mesh diameter / bisection-limited all-to-all on a 2D torus.
* f(P) = P^(1/3) — 3D torus diameter (not tabulated).

## Asymptotics
For f = log₂ P: S_C ≈ (1−α)P/(1+γ log₂ P), so E_C → **0 as 1/(γ log₂ P)** — slowly, but the 1−α floor of classical
Gustafson no longer exists. For f = √P: E_C → (1−α)/(γ√P) → 0 faster; the speedup then grows only like √P.
Weak scaling survives only if C(P) stays bounded, i.e. communication per core must be O(1) (nearest-neighbor halo
exchange with constant surface/volume, Task 17).

## Numeric example (α = 0.04, γ = 0.01 for log₂ P; γ = 0.001 for √P)

| P | E (classical) | E, C = 0.01·log₂P | E, C = 0.001·√P |
|---|---|---|---|
| 16 | 0.9625 | 0.9255 | 0.9587 |
| 64 | 0.9606 | 0.9062 | 0.9530 |
| 256 | 0.9602 | 0.8890 | 0.9450 |
| 1,024 | 0.9600 | 0.8728 | 0.9303 |
| 4,096 | 0.9600 | 0.8572 | 0.9023 |
| 16,384 | 0.9600 | 0.8421 | 0.8511 |
| 65,536 | 0.9600 | 0.8276 | 0.7643 |
| 1,048,576 | 0.9600 | 0.8000 | 0.4743 |

Reading: with only 1% per log₂-level of overhead, efficiency at one million cores is 0.80 instead of 0.96,
and the √P model drops to 0.47.
