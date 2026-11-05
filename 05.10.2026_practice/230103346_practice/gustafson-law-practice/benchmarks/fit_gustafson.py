#!/usr/bin/env python3
"""Task 23/24: summarise results.csv, extract the empirical serial fraction alpha by least squares and plot.

  python3 benchmarks/fit_gustafson.py [--csv benchmarks/results.csv] [--plotdir plots]

Weak-scaling definitions (per-thread problem size constant, total work = P * W1):
  scaled speedup  S(P) = P * T(1) / T(P)       (work done is P times larger)
  scaled efficiency E(P) = S(P)/P = T(1)/T(P)
Gustafson:  S = P - alpha (P-1)   =>   P - S = alpha (P-1)
Least squares through the origin:  alpha = sum (P-1)(P-S) / sum (P-1)^2   (P=1 carries no information)
Also reported: free-intercept fit S = a + b*P (b ~ 1-alpha when the model holds).
"""
import argparse, csv, pathlib, collections
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(HERE / "results.csv"))
ap.add_argument("--plotdir", default=str(HERE.parent / "plots"))
ap.add_argument("--no-plot", action="store_true")
a = ap.parse_args()

data, host = collections.defaultdict(list), ""
with open(a.csv) as fh:
    for r in csv.DictReader(fh):
        data[int(r["threads"])].append(float(r["time_s"])); host = r.get("host", "")
Ps = np.array(sorted(data)); assert 1 in data, "need a P=1 baseline"
mean = np.array([np.mean(data[p]) for p in Ps]); std = np.array([np.std(data[p], ddof=1) if len(data[p]) > 1 else 0.0 for p in Ps])
n = np.array([len(data[p]) for p in Ps]); se = std / np.sqrt(n)
T1, se1 = mean[0], se[0]
S = Ps * T1 / mean; E = S / Ps
S_err = S * np.sqrt((se1 / T1) ** 2 + (se / mean) ** 2)

mask = Ps > 1
x, y = (Ps[mask] - 1.0), (Ps[mask] - S[mask])
alpha = float(np.sum(x * y) / np.sum(x * x))
alpha_se = float(np.sqrt(np.sum((y - alpha * x) ** 2) / max(len(x) - 1, 1) / np.sum(x * x)))
b, a0 = np.polyfit(Ps, S, 1)
print(f"host: {host}")
print(f"{'P':>4} {'mean[s]':>10} {'std[s]':>9} {'S':>8} {'E':>7}")
for i, p in enumerate(Ps): print(f"{p:>4} {mean[i]:>10.4f} {std[i]:>9.4f} {S[i]:>8.3f} {E[i]:>7.3f}")
print(f"alpha_observed (LS through origin) = {alpha:.4f} +/- {alpha_se:.4f}")
print(f"free fit S = {a0:.3f} + {b:.3f} P  (slope ~ 1-alpha = {1-alpha:.3f})")

with open(HERE / "fit_summary.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["threads", "n_runs", "mean_s", "std_s", "S_scaled", "S_stderr", "E_scaled"])
    for i, p in enumerate(Ps): w.writerow([p, n[i], f"{mean[i]:.6f}", f"{std[i]:.6f}", f"{S[i]:.4f}", f"{S_err[i]:.4f}", f"{E[i]:.4f}"])
    w.writerow([]); w.writerow(["alpha_observed", f"{alpha:.6f}", "stderr", f"{alpha_se:.6f}"])

if not a.no_plot:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pd = pathlib.Path(a.plotdir); pd.mkdir(exist_ok=True)
    pc = np.linspace(1, Ps.max(), 200)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(pc, pc, "k:", label="ideal linear speedup")
    ax.plot(pc, pc - alpha * (pc - 1), "C0-", lw=2, label=f"Gustafson fit (α={alpha:.3f})")
    ax.plot(pc, 1 / (alpha + (1 - alpha) / pc), "C3--", lw=2, label=f"Amdahl, same s={alpha:.3f}")
    ax.errorbar(Ps, S, yerr=S_err, fmt="o", color="C1", capsize=3, label="measured (±1 s.e.)")
    ax.set_xlabel("Threads P (work ∝ P)"); ax.set_ylabel("Scaled speedup S(P)")
    ax.set_title("Weak scaling: measured vs. Gustafson vs. Amdahl\n" + host[:70]); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(pd / "scaling_comparison.png", dpi=300); fig.savefig(pd / "scaling_comparison.pdf"); plt.close(fig)
    print("plots ->", pd)
