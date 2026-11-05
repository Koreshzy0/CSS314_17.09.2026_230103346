# Task 14 — Isoefficiency of a Gustafson-Compliant Program

**Definitions (Grama–Gupta–Kumar).** $W$ = serial work (time on one core), $T_P$ = parallel time, total overhead
$T_o(W,P)=P\,T_P-W$. Efficiency $E=\dfrac{W}{W+T_o}=\dfrac{1}{1+T_o/W}$.
Constant $E$ requires $W=K\,T_o(W,P)$ with $K=E/(1-E)$; solving for $W=W(P)$ gives the isoefficiency function.

**Gustafson program.** Parallel runtime fixed $T_P=1$ with serial part $s=\alpha$ (one core busy) and parallel part
$p=1-\alpha$. Then
$$W=s+pP,\qquad T_o=P\cdot1-W=P-\alpha-(1-\alpha)P=\alpha(P-1).$$

**Efficiency:** $E=\dfrac{1}{1+\alpha(P-1)/(\alpha+(1-\alpha)P)}=\dfrac{W}{P}$ — equals $1-\alpha+\alpha/P$ (Task 04) ✓.

**Isoefficiency.** For a fixed target $E_0$: $W=K\,T_o=K\alpha(P-1)$. In Gustafson's setting
$W=\alpha+(1-\alpha)P$ already grows linearly in $P$, which satisfies $W=\Theta(P)$ with
$E\to(1-\alpha)$; to hold *exactly* $E_0<1-\alpha$ one needs only $W=\Theta(P)$ (any linear growth with the right constant).

$$\boxed{\text{isoefficiency function } \;W(P)=\Theta(P)}$$

**Critique.**
* $\Theta(P)$ is the *best possible* isoefficiency (perfectly scalable system): the problem must grow no faster than
  the machine. A program whose isoefficiency is $\Theta(P\log P)$ (e.g. with $\log P$ reductions, Task 12) or $\Theta(P^{3/2})$
  (1D-decomposed algorithms, bisection-limited comm) cannot satisfy the constant-time premise: its problem size has to
  grow *faster* than $P$ to keep $E$, so a constant-time Gustafson run loses efficiency.
* Memory limit: if memory per core is $M$, the largest problem is $W\le c\,P\,M^{\,k}$; weak scaling with
  $\Theta(P)$ fits exactly in memory when work $\propto$ memory. Algorithms with $\Theta(P^{3/2})$ isoefficiency
  (e.g. GEMM with naive 1D layout) will exceed memory before keeping $E$ — the link to Sun–Ni (Task 11).
* Contrast: under Amdahl (fixed $W$), $T_o=P\,T_P-W=\,sW(P-1)$ grows linearly in $P$ at fixed $W$, so $E\to0$ — the
  isoefficiency function says $W$ must grow ∝ $P$ to compensate; Gustafson's law is exactly that compensation.
