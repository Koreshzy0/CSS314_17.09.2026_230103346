#!/usr/bin/env python3
"""Task 21 - Amdahl (strong scaling) vs Gustafson (weak scaling) simulator.

Usage examples (from repo root):
  python3 benchmarks/simulator.py                       # defaults, writes CSV + plots
  python3 benchmarks/simulator.py --alphas 0.01 0.05 0.2 --pmax 4096 --outdir out
Domain: 1 <= P <= 65536, 0.001 <= alpha <= 0.20 (enforced).
Outputs: <outdir>/simulation.csv and <outdir>/overlay_{speedup,efficiency}.png
"""
import argparse, csv, pathlib, sys
import numpy as np

def amdahl(P, s):     return 1.0 / (s + (1.0 - s) / P)
def gustafson(P, a):  return P - a * (P - 1.0)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmin", type=int, default=1)
    ap.add_argument("--pmax", type=int, default=65536)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.001, 0.01, 0.04, 0.08, 0.20])
    ap.add_argument("--outdir", default=str(pathlib.Path(__file__).parent / "sim_output"))
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    if not (1 <= a.pmin <= a.pmax <= 65536): sys.exit("P range must satisfy 1 <= pmin <= pmax <= 65536")
    if any(not (0.001 <= x <= 0.20) for x in a.alphas): sys.exit("alpha must be in [0.001, 0.20]")

    k0, k1 = int(np.floor(np.log2(a.pmin))), int(np.ceil(np.log2(a.pmax)))
    Ps = sorted({int(x) for x in 2 ** np.arange(k0, k1 + 1) if a.pmin <= x <= a.pmax} | {a.pmin, a.pmax})
    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out / "simulation.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["P", "alpha", "S_amdahl", "E_amdahl", "S_gustafson", "E_gustafson", "ratio_G_over_A"])
        for al in a.alphas:
            for P in Ps:
                sa, sg = amdahl(P, al), gustafson(P, al)
                row = [P, al, sa, sa / P, sg, sg / P, sg / sa]; rows.append(row); w.writerow(row)
    print(f"wrote {len(rows)} rows -> {out/'simulation.csv'}")
    if a.no_plot: return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(a.alphas)))
    for key, ylabel, fname, logy in [(0, "Speedup", "overlay_speedup.png", True), (1, "Parallel efficiency", "overlay_efficiency.png", False)]:
        fig, ax = plt.subplots(figsize=(8, 5.2))
        P = np.array(Ps, dtype=float)
        ax.plot(P, P if key == 0 else np.ones_like(P), "k:", lw=1, label="ideal")
        for c, al in zip(cols, a.alphas):
            sa, sg = amdahl(P, al), gustafson(P, al)
            ax.plot(P, sg if key == 0 else sg / P, "-", color=c, lw=2, label=f"Gustafson α={al}")
            ax.plot(P, sa if key == 0 else sa / P, "--", color=c, lw=2, label=f"Amdahl s={al}")
        ax.set_xscale("log", base=2)
        if logy: ax.set_yscale("log", base=2)
        ax.set_xlabel("Processors P"); ax.set_ylabel(ylabel); ax.grid(True, which="both", alpha=.3)
        ax.set_title(f"{ylabel}: strong (Amdahl, dashed) vs weak (Gustafson, solid) scaling")
        ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(out / fname, dpi=200); plt.close(fig)
    print("plots ->", out)

if __name__ == "__main__":
    main()
