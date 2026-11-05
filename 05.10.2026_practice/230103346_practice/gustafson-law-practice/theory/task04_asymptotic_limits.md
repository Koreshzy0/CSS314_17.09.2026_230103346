# Task 04 — Asymptotic Parallel Efficiency

## Gustafson
$$E_G(P)=\frac{P-\alpha(P-1)}{P}=1-\alpha+\frac{\alpha}{P}.$$

**Limit.** $\displaystyle\lim_{P\to\infty}E_G(P)=1-\alpha.$

*Proof (ε–N).* $|E_G(P)-(1-\alpha)|=\alpha/P<\varepsilon$ whenever $P>\alpha/\varepsilon$. ∎
$E_G'(P)=-\alpha/P^2<0$, so $E_G$ decreases monotonically from $E_G(1)=1$ to the floor $1-\alpha$.

## Amdahl
$$E_A(P)=\frac{S_A}{P}=\frac{1}{sP+(1-s)}\ \Rightarrow\ \lim_{P\to\infty}E_A(P)=0\quad(s>0),\qquad E_A\sim\frac1{sP}.$$
Efficiency is 50% already at $P=1+1/s$ (for $s=0.04$: $P=26$).

## Contrast ($\alpha=s=0.04$)

| | $P=16$ | $P=1024$ | $P\to\infty$ |
|---|---|---|---|
| Gustafson $E$ | 0.9625 | 0.9600 | **0.96** |
| Amdahl $E$ | 0.625 | 0.0244 | **0** |

**Deduction.** Under weak scaling the machine stays usefully busy for a constant fraction $1-\alpha$; under strong
scaling the same serial fraction wastes an ever larger share of the machine, in the limit all of it.
