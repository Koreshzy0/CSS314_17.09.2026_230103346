# Task 15 — Energy Under Gustafson Scaling

**Model.** Each node draws static power $P_{idle}$ whenever the job holds it, plus dynamic power $P_{act}$ when its
core is computing (so a computing node draws $P_{idle}+P_{act}$). Wall-clock time $T$ (the same on 1 and $P$ cores,
fixed-time premise). Serial fraction $\alpha=s$, $p=1-\alpha$.

## Energy on $P$ cores (scaled workload)
* Static: all $P$ nodes allocated for $T$: $\;E_{static}=P\,P_{idle}\,T$.
* Dynamic: busy core-time = serial $sT$ (one core) + parallel $pT\cdot P$:
  $\;E_{dyn}=P_{act}\,T\,(s+pP)$.

$$E_P = T\left[P\,P_{idle}+P_{act}\,(\alpha+(1-\alpha)P)\right]=T\left[P\,P_{idle}+P_{act}\,S(P)\right].$$

## Energy on 1 core, identical wall-clock $T$
Only the *unscaled* workload fits: $E_1(T)=T(P_{idle}+P_{act})$ (it does $1/S(P)$ as much work).

## Fair comparison: same scaled workload on 1 core
Time $T_1 = S(P)\,T$, energy $E^{ser}=S(P)\,T\,(P_{idle}+P_{act})$.

$$E_P-E^{ser}=T\,P_{idle}\,[P-S(P)]=T\,P_{idle}\,\alpha\,(P-1).$$

**Result:** the parallel run uses *extra* energy exactly equal to idle power of the $P-1$ cores waiting during the serial
section. The dynamic energy is identical (same number of operations). If $\alpha=0$ both consume the same energy but
the parallel run finishes $S$ times sooner.

## Energy per unit of work
$$\varepsilon_P=\frac{E_P}{S(P)T}=P_{act}+P_{idle}\frac{P}{S(P)}=P_{act}+\frac{P_{idle}}{E(P)}.$$
Energy per unit work rises with $1/E(P)$ — the energy overhead factor of weak scaling is the inverse of parallel efficiency
on the static-power term only.

## Numeric example
$P_{idle}=50$ W, $P_{act}=150$ W per core, $\alpha=0.04$, $P=64$, $T=1$ h:
* $S=61.48$; $E_P=64\cdot50+150\cdot61.48=3200+9222=12{,}422$ Wh.
* 1 core, same wall time: $200$ Wh (does 1/61.48 of the work).
* 1 core, same scaled workload: $200\cdot61.48=12{,}296$ Wh (takes 61.48 h).
* Difference $=50\cdot0.04\cdot63=126$ Wh ✓ ($+1.02\%$).

**Conclusion.** Under Gustafson scaling, energy-to-solution is within $P_{idle}\alpha(P-1)/(\dots)$ of the serial
energy; the dominating benefit is time. If idle power is high relative to active power (real servers: 40–60%), energy
efficiency tracks parallel efficiency, so $\alpha(P)$ growth (Task 12) hurts energy as well as time. Power-gating idle
cores during serial phases reduces the penalty.
